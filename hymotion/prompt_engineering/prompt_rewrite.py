# prompt_rewrite.py
import base64
import concurrent.futures
import datetime
import hashlib
import hmac
import json
import logging
import random
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
from openai import OpenAI
from requests import exceptions as req_exc
from transformers import AutoModelForCausalLM, AutoTokenizer

from .model_constants import REWRITE_AND_INFER_TIME_PROMPT_FORMAT

# logging.basicConfig(level=logging.INFO)


@dataclass
class ApiConfig:
    host: str
    user: str
    apikey: str
    model: str
    api_version: Optional[str] = None
    timeout: int = 3600
    source: str = "hymotion"


@dataclass
class RetryConfig:
    max_retries: int = 20
    base_delay: float = 1.0
    timeout: float = 30.0
    retry_status: Tuple[int, ...] = (429, 500, 502, 503, 504)
    max_delay: float = 1.0


class ApiError(Exception):
    pass


class ResponseParseError(Exception):
    pass


class OpenAIChatApi:
    def __init__(self, config: ApiConfig) -> None:
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.client = OpenAI(
            api_key=self.config.apikey,
            base_url=self.config.host,
        )

    def call_data_eval(self, data: Union[str, Dict[str, Any]]):
        if isinstance(data, dict) and "messages" in data:
            raw_msgs = data["messages"]
            messages: List[Dict[str, str]] = []
            for m in raw_msgs:
                role = m.get("role", "user")
                content = m.get("content", "")
                if isinstance(content, list):
                    parts = []
                    for p in content:
                        if isinstance(p, dict) and ("text" in p):
                            parts.append(str(p.get("text", "")))
                    content = " ".join([t for t in parts if t])
                elif not isinstance(content, str):
                    content = str(content)
                messages.append({"role": role, "content": content})
            payload = {"model": self.config.model, "messages": messages}
            for k in (
                "temperature",
                "top_p",
                "max_tokens",
                "n",
                "stop",
                "presence_penalty",
                "frequency_penalty",
                "user",
            ):
                if k in data:
                    payload[k] = data[k]
        else:
            payload = {"model": self.config.model, "messages": [{"role": "user", "content": str(data)}]}
        try:
            resp = self.client.chat.completions.create(**payload)
            return resp
        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {e}")
            raise ApiError(f"OpenAI API call failed: {e}") from e


class ResponseParser:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def call_data_eval_with_retry(
        self, api: Union[OpenAIChatApi], data: str, retry_config: Optional[RetryConfig] = None
    ) -> Tuple[Union[Dict[str, Any], int], float, float]:
        if retry_config is None:
            retry_config = RetryConfig()

        last_error = None
        for attempt in range(retry_config.max_retries):
            start_time = time.time()
            cost = 0.0

            try:
                result = self._execute_request(api, data)
                end_time = time.time()
                parsed_result = self._parse_answer(result)
                self._validate_result(parsed_result)
                return parsed_result, cost, end_time - start_time

            except (
                concurrent.futures.TimeoutError,
                req_exc.RequestException,
                json.JSONDecodeError,
                ValueError,
                TypeError,
                ResponseParseError,
            ) as e:
                last_error = e
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if isinstance(e, req_exc.RequestException) and hasattr(e, "response"):
                    if e.response is not None and e.response.status_code not in retry_config.retry_status:
                        raise ApiError(f"Non-retryable error: {e.response.status_code}") from e
                if attempt < retry_config.max_retries - 1:
                    delay = self._calculate_delay(attempt, retry_config)
                    self.logger.info(f"JSON parsing failed, {delay:.1f} seconds later retry...")
                    time.sleep(delay)

        raise ApiError(f"Retry {retry_config.max_retries} times but still failed") from last_error

    def _execute_request(self, api: Union[OpenAIChatApi], data: str) -> Dict[str, Any]:
        response = api.call_data_eval(data)

        try:
            if hasattr(response, "model_dump"):
                return response.model_dump()
            if isinstance(response, dict):
                return response
            if hasattr(response, "__dict__"):
                return json.loads(json.dumps(response.__dict__, default=str))
        except Exception as e:
            raise ResponseParseError(f"Unable to parse OpenAI returned object: {type(response)} - {e}") from e

        raise ResponseParseError(f"Unknown response type: {type(response)}")

    def _extract_cost(self, payload: Dict[str, Any]) -> float:
        try:
            return float(payload.get("cost_info", {}).get("cost", 0)) / 1e6
        except (AttributeError, KeyError):
            return 0.0

    def _validate_result(self, result: Union[Dict[str, Any], int]) -> None:
        if isinstance(result, int):
            return
        elif isinstance(result, dict):
            required_fields = ["duration", "short_caption"]
            for field in required_fields:
                if not isinstance(result.get(field), (int, str)):
                    raise ResponseParseError(f"LLM returned invalid format: {field}")
        else:
            raise ResponseParseError(f"Unsupported answer type: {type(result)}")

    def _calculate_delay(self, attempt: int, config: RetryConfig) -> float:
        delay = config.base_delay * (2**attempt) * (0.5 + random.random())
        return min(delay, config.max_delay)

    def _parse_answer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(payload, dict) and "choices" in payload:
            return self._parse_from_choices_field(payload)

        raise ResponseParseError("Unknown response format: expected choices")

    def _parse_from_choices_field(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        choices = payload.get("choices") or []
        if not choices:
            raise ResponseParseError("OpenAI returned empty")

        content = self._extract_content_from_choice(choices[0])

        if not isinstance(content, str) or not content.strip():
            raise ResponseParseError("OpenAI returned no valid content")

        return self._parse_json_content(content)

    def _extract_content_from_choice(self, choice: Any) -> Optional[str]:
        content = None

        if isinstance(choice, dict):
            # Try message content first
            msg = choice.get("message") or {}
            content = msg.get("content")
            # Fallback to delta content or text
            if content is None:
                delta = choice.get("delta") or {}
                content = delta.get("content", choice.get("text"))
        else:
            # Handle object-like choice (e.g. Pydantic model)
            msg = getattr(choice, "message", None)
            if msg is not None:
                content = getattr(msg, "content", None)

            if content is None:
                delta = getattr(choice, "delta", None)
                if delta is not None:
                    content = getattr(delta, "content", None)

            if content is None:
                content = getattr(choice, "text", None)

        return content

    def _parse_json_content(self, content: str) -> Dict[str, Any]:
        cleaned = self._cleanup_fenced_json(content)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON parsing failed, original content: {cleaned[:500]}...")
            raise ResponseParseError(f"JSON parsing failed: {e}") from e

    def _cleanup_fenced_json(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        if not text.lstrip().startswith("{") and "{" in text and "}" in text:
            start = text.find("{")
            end = text.rfind("}")
            if 0 <= start < end:
                text = text[start : end + 1]
        return text


class PromptRewriter:
    def __init__(
        self, host: Optional[str] = None, model_path: Optional[str] = None, parser: Optional[ResponseParser] = None,
        offload_to_cpu: bool = False
    ):
        self.parser = parser or ResponseParser()
        self.logger = logging.getLogger(__name__)
        self.host = host
        self.offload_to_cpu = offload_to_cpu
        if host:
            self.api = OpenAIChatApi(
                ApiConfig(
                    host=host,
                    user="",
                    apikey="EMPTY",
                    model="Qwen3-30B-A3B-SFT",
                    api_version="",
                )
            )
        else:
            self.model_path = model_path or "Text2MotionPrompter/Text2MotionPrompter"
            self.tokenizer = None
            self.model = None
            self._load_model()

    def _load_model(self):
        if self.model is None:
            print(f">>> Loading prompter model from {self.model_path}, offload_to_cpu={self.offload_to_cpu}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

            if self.offload_to_cpu:
                # Load entirely on CPU (slower but saves GPU memory)
                print(">>> Loading prompter on CPU...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                )
            else:
                # Load on GPU with 4bit quantization
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    quantization_config=quantization_config,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                )

            self.model.eval()

    def rewrite_prompt_and_infer_time(
        self,
        text: str,
        prompt_format: str = REWRITE_AND_INFER_TIME_PROMPT_FORMAT,
        retry_config: Optional[RetryConfig] = None,
    ) -> Tuple[float, str]:
        if self.host:
            self.logger.info("Start rewriting prompt...")
            try:
                result, cost, elapsed = self.parser.call_data_eval_with_retry(
                    self.api, prompt_format.format(text), retry_config
                )
                self.logger.info(f"Rewriting completed - cost: {cost:.6f}, time: {elapsed:.2f}s")
                return round(float(result["duration"]) / 30.0, 2), result["short_caption"]

            except Exception as e:
                self.logger.error(f"Prompt rewriting failed: {e}")
                raise
        else:
            messages = [{"role": "user", "content": prompt_format.format(text)}]
            full_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            # Get device from model parameters
            device = next(self.model.parameters()).device
            inputs = self.tokenizer([full_prompt], return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=8192)
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1] :].tolist(), skip_special_tokens=True)

            print(f">>> [Prompter] Raw LLM response:\n{response[:500]}")

            try:
                json_str = re.search(r"\{.*\}", response, re.DOTALL).group()
                print(f">>> [Prompter] Extracted JSON: {json_str}")
                result = json.loads(json_str)
                duration = round(float(result["duration"]) / 30.0, 2)
                short_caption = result["short_caption"]
                print(f">>> [Prompter] Parsed - duration: {result['duration']} frames, caption: {short_caption}")
                return duration, short_caption
            except Exception as e:
                print(f">>> [Prompter] JSON parse failed: {e}, returning original text")
                return 5.0, text


if __name__ == "__main__":
    # python -m hymotion.prompt_engineering.prompt_rewrite

    logging.basicConfig(level=logging.INFO)
    text = "person jumps after they runs"
    prompt_rewriter = PromptRewriter()
    result = prompt_rewriter.rewrite_prompt_and_infer_time(text)
    print(result)

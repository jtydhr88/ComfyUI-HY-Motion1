import os
import sys
import uuid
import time
import numpy as np
import torch
from typing import Optional, List, Dict, Any

import comfy.model_management as model_management

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

try:
    import folder_paths
    COMFY_MODELS_DIR = folder_paths.models_dir
    COMFY_OUTPUT_DIR = folder_paths.get_output_directory()
except ImportError:
    COMFY_MODELS_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "models")
    COMFY_OUTPUT_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "output")

HYMOTION_MODELS_DIR = os.path.join(COMFY_MODELS_DIR, "HY-Motion")


def get_timestamp():
    t = time.time()
    ms = int((t - int(t)) * 1000)
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(t)) + f"{ms:03d}"


class HYMotionLLMWrapper:
    """LLM model wrapper"""
    def __init__(self, model, tokenizer, llm_type="qwen3", max_length=512, crop_start=0):
        self.model = model
        self.tokenizer = tokenizer
        self.llm_type = llm_type
        self.max_length = max_length
        self.crop_start = crop_start
        self.hidden_size = model.config.hidden_size if hasattr(model, 'config') else 4096


class HYMotionNetworkWrapper:
    """Diffusion Network wrapper"""
    def __init__(self, network, config, mean, std, body_model=None):
        self.network = network
        self.config = config
        self.mean = mean
        self.std = std
        self.body_model = body_model


class HYMotionConditioning:
    """Text encoding result"""
    def __init__(self, vtxt_raw, ctxt_raw, ctxt_length, text: List[str]):
        self.vtxt_raw = vtxt_raw
        self.ctxt_raw = ctxt_raw
        self.ctxt_length = ctxt_length
        self.text = text

    def to_hidden_state_dict(self):
        return {
            "text_vec_raw": self.vtxt_raw,
            "text_ctxt_raw": self.ctxt_raw,
            "text_ctxt_raw_length": self.ctxt_length,
        }


class HYMotionData:
    """Motion output data"""
    def __init__(self, output_dict: Dict[str, Any], text: str, duration: float, seeds: List[int]):
        self.output_dict = output_dict
        self.text = text
        self.duration = duration
        self.seeds = seeds
        self.batch_size = output_dict["keypoints3d"].shape[0] if "keypoints3d" in output_dict else 1


class HYMotionPrompterWrapper:
    """Prompt Rewriter model wrapper"""
    def __init__(self, rewriter):
        self.rewriter = rewriter


# ============================================================================
# Node 1a: HYMotion Load LLM (HuggingFace)
# ============================================================================

class HYMotionLoadLLM:
    """Load Qwen3 LLM with quantization support"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "quantization": (["none", "int8", "int4"], {"default": "none"}),
                "offload_to_cpu": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("HYMOTION_LLM",)
    RETURN_NAMES = ("llm",)
    FUNCTION = "load_llm"
    CATEGORY = "HY-Motion/Loaders"

    def load_llm(self, quantization="none", offload_to_cpu=False):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from .hymotion.network.text_encoders.model_constants import PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION

        local_path = os.path.join(HYMOTION_MODELS_DIR, "ckpts", "Qwen3-8B")
        model_path = local_path if os.path.exists(local_path) else "Qwen/Qwen3-8B"

        print(f"[HY-Motion] Loading LLM: {model_path}, quantization={quantization}, offload_to_cpu={offload_to_cpu}")

        load_kwargs = {"low_cpu_mem_usage": True}

        if offload_to_cpu:
            load_kwargs["device_map"] = "cpu"
            load_kwargs["torch_dtype"] = torch.float32
        elif quantization == "int8":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs["device_map"] = "auto"
        elif quantization == "int4":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["device_map"] = "auto"

        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        model = model.eval().requires_grad_(False)

        # Compute crop_start
        template = [
            {"role": "system", "content": f"{PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION}"},
            {"role": "user", "content": "{}"},
        ]
        crop_start = self._compute_crop_start(tokenizer, template)

        if not offload_to_cpu and quantization == "none":
            device = model_management.get_torch_device()
            model = model.to(device)

        wrapper = HYMotionLLMWrapper(model=model, tokenizer=tokenizer, max_length=512, crop_start=crop_start)
        print(f"[HY-Motion] LLM loaded, hidden_size={wrapper.hidden_size}")
        return (wrapper,)

    def _compute_crop_start(self, tokenizer, template) -> int:
        def _find_subseq(a, b):
            for i in range(len(a) - len(b) + 1):
                if a[i:i + len(b)] == b:
                    return i
            return -1

        marker = "<BOC>"
        msgs = [{"role": "system", "content": template[0]['content']}, {"role": "user", "content": marker}]
        s = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        full_ids = tokenizer(s, return_tensors="pt", add_special_tokens=True)["input_ids"][0].tolist()
        marker_ids = tokenizer(marker, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        pos = _find_subseq(full_ids, marker_ids)
        return pos if pos >= 0 else max(0, len(full_ids) - 1)


# ============================================================================
# Node 1b: HYMotion Load LLM GGUF
# ============================================================================

class HYMotionLoadLLMGGUF:
    """Load Qwen3 LLM from GGUF file

    Supports loading GGUF quantized models via:
    1. transformers native GGUF support (recommended, requires transformers>=4.40)

    Place GGUF files in: ComfyUI/models/HY-Motion/ckpts/GGUF/
    """

    @classmethod
    def INPUT_TYPES(s):
        # Scan for GGUF files
        gguf_files = ["(select file)"]
        gguf_dir = os.path.join(HYMOTION_MODELS_DIR, "ckpts", "GGUF")
        if os.path.exists(gguf_dir):
            for f in os.listdir(gguf_dir):
                if f.endswith(".gguf"):
                    gguf_files.append(f)

        llm_gguf_dir = os.path.join(COMFY_MODELS_DIR, "llm", "GGUF")
        if os.path.exists(llm_gguf_dir):
            for f in os.listdir(llm_gguf_dir):
                if f.endswith(".gguf") and "qwen" in f.lower():
                    gguf_files.append(f"llm/GGUF/{f}")

        return {
            "required": {
                "gguf_file": (gguf_files, {"default": "(select file)"}),
                "offload_to_cpu": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("HYMOTION_LLM",)
    RETURN_NAMES = ("llm",)
    FUNCTION = "load_llm_gguf"
    CATEGORY = "HY-Motion/Loaders"

    def load_llm_gguf(self, gguf_file, offload_to_cpu=False):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .hymotion.network.text_encoders.model_constants import PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION

        # Determine the actual path
        if gguf_file == "(select file)":
            raise ValueError("Please select a GGUF file")

        if gguf_file.startswith("llm/GGUF/"):
            gguf_dir = os.path.join(COMFY_MODELS_DIR, "llm", "GGUF")
            gguf_filename = os.path.basename(gguf_file)
        else:
            gguf_dir = os.path.join(HYMOTION_MODELS_DIR, "ckpts", "GGUF")
            gguf_filename = gguf_file

        gguf_path = os.path.join(gguf_dir, gguf_filename)
        if not os.path.exists(gguf_path):
            raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

        print(f"[HY-Motion] Loading LLM from GGUF: {gguf_path}, offload_to_cpu={offload_to_cpu}")

        # Check transformers version for native GGUF support
        try:
            import transformers
            tf_version = tuple(map(int, transformers.__version__.split('.')[:2]))
            has_native_gguf = tf_version >= (4, 40)
        except:
            has_native_gguf = False

        if has_native_gguf:
            tokenizer_path = os.path.join(HYMOTION_MODELS_DIR, "ckpts", "Qwen3-8B")
            if not os.path.exists(tokenizer_path):
                tokenizer_path = "Qwen/Qwen3-8B"

            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side="right")

            load_kwargs = {
                "gguf_file": gguf_filename,
                "low_cpu_mem_usage": True,
            }
            if offload_to_cpu:
                load_kwargs["device_map"] = "cpu"
                load_kwargs["torch_dtype"] = torch.float32
            else:
                load_kwargs["device_map"] = "auto"

            model = AutoModelForCausalLM.from_pretrained(gguf_dir, **load_kwargs)
            model = model.eval().requires_grad_(False)
        else:
            print("[HY-Motion] transformers<4.40, GGUF not supported")
            raise NotImplementedError(
                "GGUF support requires transformers>=4.40. "
                "Please upgrade: pip install -U transformers>=4.40"
            )

        # Compute crop_start
        template = [
            {"role": "system", "content": f"{PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION}"},
            {"role": "user", "content": "{}"},
        ]
        crop_start = self._compute_crop_start(tokenizer, template)

        wrapper = HYMotionLLMWrapper(
            model=model,
            tokenizer=tokenizer,
            llm_type="qwen3_gguf",
            max_length=512,
            crop_start=crop_start
        )
        print(f"[HY-Motion] GGUF LLM loaded, hidden_size={wrapper.hidden_size}")
        return (wrapper,)

    def _compute_crop_start(self, tokenizer, template) -> int:
        def _find_subseq(a, b):
            for i in range(len(a) - len(b) + 1):
                if a[i:i + len(b)] == b:
                    return i
            return -1

        marker = "<BOC>"
        msgs = [{"role": "system", "content": template[0]['content']}, {"role": "user", "content": marker}]
        try:
            s = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        except TypeError:
            # Some tokenizers don't support enable_thinking
            s = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        full_ids = tokenizer(s, return_tensors="pt", add_special_tokens=True)["input_ids"][0].tolist()
        marker_ids = tokenizer(marker, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        pos = _find_subseq(full_ids, marker_ids)
        return pos if pos >= 0 else max(0, len(full_ids) - 1)


# ============================================================================
# Node 2: HYMotion Load Network
# ============================================================================

class HYMotionLoadNetwork:
    """Load Motion Diffusion Network"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (["HY-Motion-1.0", "HY-Motion-1.0-Lite"], {"default": "HY-Motion-1.0-Lite"}),
            },
        }

    RETURN_TYPES = ("HYMOTION_NET",)
    RETURN_NAMES = ("network",)
    FUNCTION = "load_network"
    CATEGORY = "HY-Motion/Loaders"

    def load_network(self, model_name):
        import yaml
        from .hymotion.utils.loaders import load_object
        from .hymotion.pipeline.body_model import WoodenMesh

        model_dir = os.path.join(HYMOTION_MODELS_DIR, "ckpts", "tencent", model_name)
        config_path = os.path.join(model_dir, "config.yml")
        ckpt_path = os.path.join(model_dir, "latest.ckpt")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")

        print(f"[HY-Motion] Loading network: {config_path}")

        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        network = load_object(config["network_module"], config["network_module_args"])
        network.eval()

        # Load weights
        mean = torch.zeros(201)
        std = torch.ones(201)
        null_vtxt_feat = torch.randn(1, 1, 768)
        null_ctxt_input = torch.randn(1, 1, 4096)

        if os.path.exists(ckpt_path):
            print(f"[HY-Motion] Loading checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = checkpoint["model_state_dict"]

            network_state = {k.replace("motion_transformer.", ""): v
                           for k, v in state_dict.items() if k.startswith("motion_transformer.")}
            network.load_state_dict(network_state, strict=False)

            mean = state_dict.get("mean", mean)
            std = state_dict.get("std", std)
            null_vtxt_feat = state_dict.get("null_vtxt_feat", null_vtxt_feat)
            null_ctxt_input = state_dict.get("null_ctxt_input", null_ctxt_input)

        # Load from stats directory
        stats_dir = os.path.join(model_dir, "stats")
        if not os.path.exists(stats_dir):
            stats_dir = os.path.join(CURRENT_DIR, "HY-Motion-1.0", "stats")
        if os.path.exists(stats_dir):
            mean_path = os.path.join(stats_dir, "Mean.npy")
            std_path = os.path.join(stats_dir, "Std.npy")
            if os.path.exists(mean_path) and os.path.exists(std_path):
                mean = torch.from_numpy(np.load(mean_path)).float()
                std = torch.from_numpy(np.load(std_path)).float()

        device = model_management.get_torch_device()
        network = network.to(device)
        mean = mean.to(device)
        std = std.to(device)

        wrapper = HYMotionNetworkWrapper(network=network, config=config, mean=mean, std=std, body_model=WoodenMesh())
        wrapper.train_frames = 360
        wrapper.output_mesh_fps = 30
        wrapper.validation_steps = 50
        wrapper.input_dim = config["network_module_args"].get("input_dim", 201)
        wrapper.null_vtxt_feat = null_vtxt_feat.to(device)
        wrapper.null_ctxt_input = null_ctxt_input.to(device)

        print("[HY-Motion] Network loaded")
        return (wrapper,)


# ============================================================================
# Node 2b: HYMotion Load Prompter
# ============================================================================

class HYMotionLoadPrompter:
    """Load Text2MotionPrompter LLM for prompt rewriting and duration estimation"""

    @classmethod
    def INPUT_TYPES(s):
        # Scan for local prompter models
        prompter_models = ["(auto download)"]
        prompter_dir = os.path.join(HYMOTION_MODELS_DIR, "ckpts", "Text2MotionPrompter")
        if os.path.exists(prompter_dir):
            prompter_models.append("local: Text2MotionPrompter")

        return {
            "required": {
                "model_source": (prompter_models, {"default": "(auto download)"}),
                "offload_to_cpu": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("HYMOTION_PROMPTER",)
    RETURN_NAMES = ("prompter",)
    FUNCTION = "load_prompter"
    CATEGORY = "HY-Motion/Loaders"

    def load_prompter(self, model_source, offload_to_cpu=False):
        from .hymotion.prompt_engineering.prompt_rewrite import PromptRewriter

        if model_source == "local: Text2MotionPrompter":
            model_path = os.path.join(HYMOTION_MODELS_DIR, "ckpts", "Text2MotionPrompter")
        else:
            # Auto download from HuggingFace
            model_path = "Text2MotionPrompter/Text2MotionPrompter"

        print(f"[HY-Motion] Loading Prompter: {model_path}, offload_to_cpu={offload_to_cpu}")
        rewriter = PromptRewriter(model_path=model_path, offload_to_cpu=offload_to_cpu)
        print(f"[HY-Motion] Prompter loaded")

        return (HYMotionPrompterWrapper(rewriter),)


# ============================================================================
# Node 2c: HYMotion Rewrite Prompt
# ============================================================================

class HYMotionRewritePrompt:
    """Rewrite text prompt and estimate motion duration using LLM"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompter": ("HYMOTION_PROMPTER",),
                "text": ("STRING", {"default": "A person is walking forward.", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT")
    RETURN_NAMES = ("rewritten_text", "duration")
    FUNCTION = "rewrite"
    CATEGORY = "HY-Motion/Conditioning"

    def rewrite(self, prompter: HYMotionPrompterWrapper, text: str):
        print(f"[HY-Motion] ========== Prompt Rewrite ==========")
        print(f"[HY-Motion] INPUT:  {text}")
        duration, rewritten_text = prompter.rewriter.rewrite_prompt_and_infer_time(text)
        print(f"[HY-Motion] OUTPUT: {rewritten_text}")
        print(f"[HY-Motion] DURATION: {duration:.2f}s ({int(duration * 30)} frames)")
        print(f"[HY-Motion] =====================================")
        return (rewritten_text, duration)


# ============================================================================
# Node 3: HYMotion Encode Text
# ============================================================================

class HYMotionEncodeText:
    """Encode text - CLIP loaded internally, LLM passed externally"""
    _clip_model = None
    _clip_tokenizer = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "llm": ("HYMOTION_LLM",),
                "text": ("STRING", {"default": "A person is walking forward.", "multiline": True}),
            },
        }

    RETURN_TYPES = ("HYMOTION_COND",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "HY-Motion/Conditioning"

    def _ensure_clip_loaded(self):
        """Lazy load CLIP model"""
        if HYMotionEncodeText._clip_model is None:
            from transformers import CLIPTextModel, CLIPTokenizer
            print("[HY-Motion] Loading CLIP (clip-vit-large-patch14)")

            local_path = os.path.join(HYMOTION_MODELS_DIR, "ckpts", "clip-vit-large-patch14")
            clip_path = local_path if os.path.exists(local_path) else "openai/clip-vit-large-patch14"

            HYMotionEncodeText._clip_tokenizer = CLIPTokenizer.from_pretrained(clip_path, max_length=77)
            HYMotionEncodeText._clip_model = CLIPTextModel.from_pretrained(clip_path)
            HYMotionEncodeText._clip_model = HYMotionEncodeText._clip_model.eval().requires_grad_(False)

            device = model_management.get_torch_device()
            HYMotionEncodeText._clip_model = HYMotionEncodeText._clip_model.to(device)
            print("[HY-Motion] CLIP loaded")

    def encode(self, llm: HYMotionLLMWrapper, text: str):
        from .hymotion.network.text_encoders.model_constants import PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION

        self._ensure_clip_loaded()

        text_list = [text]
        device = model_management.get_torch_device()

        # CLIP encoding
        enc = HYMotionEncodeText._clip_tokenizer(text_list, truncation=True, max_length=77, padding=True, return_tensors="pt")
        clip_device = next(HYMotionEncodeText._clip_model.parameters()).device
        out = HYMotionEncodeText._clip_model(input_ids=enc["input_ids"].to(clip_device), attention_mask=enc["attention_mask"].to(clip_device))
        vtxt_raw = out.pooler_output.unsqueeze(1) if out.pooler_output is not None else out.last_hidden_state.mean(1, keepdim=True)
        vtxt_raw = vtxt_raw.to(device)

        # LLM encoding
        template = [{"role": "system", "content": PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION}, {"role": "user", "content": "{}"}]
        llm_text = [llm.tokenizer.apply_chat_template(
            [{"role": "system", "content": template[0]['content']}, {"role": "user", "content": t}],
            tokenize=False, add_generation_prompt=False, enable_thinking=False
        ) for t in text_list]

        max_length = llm.max_length + llm.crop_start
        llm_enc = llm.tokenizer(llm_text, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
        llm_device = next(llm.model.parameters()).device

        llm_out = llm.model(
            input_ids=llm_enc["input_ids"].to(llm_device),
            attention_mask=llm_enc["attention_mask"].to(llm_device),
            output_hidden_states=True,
        )

        ctxt_raw = llm_out.hidden_states[-1][:, llm.crop_start:llm.crop_start + llm.max_length].contiguous().to(device)
        ctxt_length = (llm_enc["attention_mask"].sum(dim=-1) - llm.crop_start).clamp(0, llm.max_length).to(device)

        print(f"[HY-Motion] Encoded: vtxt={vtxt_raw.shape}, ctxt={ctxt_raw.shape}")
        return (HYMotionConditioning(vtxt_raw, ctxt_raw, ctxt_length, text_list),)


# ============================================================================
# Node 4: HYMotion Generate
# ============================================================================

class HYMotionGenerate:
    """Motion generation"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "network": ("HYMOTION_NET",),
                "conditioning": ("HYMOTION_COND",),
                "duration": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 12.0, "step": 0.1}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0x7fffffff}),
            },
            "optional": {
                "cfg_scale": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 15.0, "step": 0.5}),
                "num_samples": ("INT", {"default": 1, "min": 1, "max": 4}),
            }
        }

    RETURN_TYPES = ("HYMOTION_DATA",)
    RETURN_NAMES = ("motion_data",)
    FUNCTION = "generate"
    CATEGORY = "HY-Motion"

    def generate(self, network: HYMotionNetworkWrapper, conditioning: HYMotionConditioning,
                 duration: float, seed: int, cfg_scale: float = 5.0, num_samples: int = 1):
        import comfy.utils
        from torchdiffeq import odeint
        from .hymotion.pipeline.motion_diffusion import length_to_mask, randn_tensor

        device = model_management.get_torch_device()
        network.network = network.network.to(device)

        length = min(max(int(duration * network.output_mesh_fps), 20), network.train_frames)
        seeds = [seed + i for i in range(num_samples)]

        print(f"[HY-Motion] Generating: {duration}s, length={length}, seeds={seeds}")

        vtxt_input = conditioning.vtxt_raw.to(device)
        ctxt_input = conditioning.ctxt_raw.to(device)
        ctxt_length = conditioning.ctxt_length.to(device)

        if vtxt_input.shape[0] == 1 and num_samples > 1:
            vtxt_input = vtxt_input.repeat(num_samples, 1, 1)
            ctxt_input = ctxt_input.repeat(num_samples, 1, 1)
            ctxt_length = ctxt_length.repeat(num_samples)

        ctxt_mask = length_to_mask(ctxt_length, ctxt_input.shape[1])
        x_length = torch.LongTensor([length] * num_samples).to(device)
        x_mask = length_to_mask(x_length, network.train_frames)

        do_cfg = cfg_scale > 1.0
        if do_cfg:
            vtxt_input = torch.cat([network.null_vtxt_feat.expand_as(vtxt_input), vtxt_input], dim=0)
            ctxt_input = torch.cat([network.null_ctxt_input.expand_as(ctxt_input), ctxt_input], dim=0)
            ctxt_mask = torch.cat([ctxt_mask] * 2, dim=0)
            x_mask = torch.cat([x_mask] * 2, dim=0)

        pbar = comfy.utils.ProgressBar(network.validation_steps)
        step = [0]

        def fn(t, x):
            x_in = torch.cat([x] * 2, dim=0) if do_cfg else x
            pred = network.network(x=x_in, ctxt_input=ctxt_input, vtxt_input=vtxt_input,
                                   timesteps=t.expand(x_in.shape[0]), x_mask_temporal=x_mask, ctxt_mask_temporal=ctxt_mask)
            if do_cfg:
                pred_u, pred_c = pred.chunk(2)
                pred = pred_u + cfg_scale * (pred_c - pred_u)
            step[0] += 1
            pbar.update_absolute(step[0], network.validation_steps)
            return pred

        t = torch.linspace(0, 1, network.validation_steps + 1, device=device)
        y0 = torch.cat([randn_tensor((1, network.train_frames, network.input_dim),
                       generator=torch.Generator(device=device).manual_seed(s), device=device) for s in seeds])

        with torch.no_grad():
            sampled = odeint(fn, y0, t, method="euler")[-1][:, :length]

        output_dict = self._decode(sampled, network)
        print(f"[HY-Motion] Done")
        return (HYMotionData(output_dict, conditioning.text[0], duration, seeds),)

    def _decode(self, latent, net):
        from scipy.signal import savgol_filter
        from .hymotion.utils.geometry import rot6d_to_rotation_matrix, rotation_matrix_to_rot6d, matrix_to_quaternion, quaternion_to_matrix, quaternion_fix_continuity
        from .hymotion.utils.motion_process import smooth_rotation

        std = torch.where(net.std < 1e-3, torch.ones_like(net.std), net.std)
        x = latent * std + net.mean
        B, L = x.shape[:2]

        transl = x[..., :3].clone()
        rot6d = torch.cat([x[..., 3:9].reshape(B, L, 1, 6), x[..., 9:9+126].reshape(B, L, 21, 6)], dim=2)

        # Smooth rotations
        RR = rot6d_to_rotation_matrix(rot6d)
        qq = matrix_to_quaternion(RR)
        qq = qq.moveaxis(1, 0).contiguous().view(L, -1, 4)
        qq = quaternion_fix_continuity(qq).view(L, B, 22, 4).moveaxis(0, 1)
        rot6d_s = rotation_matrix_to_rot6d(quaternion_to_matrix(torch.from_numpy(smooth_rotation(qq.cpu().numpy(), sigma=1.0)))).to(latent.device)

        transl_np = transl.cpu().numpy()
        for b in range(B):
            for j in range(3):
                transl_np[b, :, j] = savgol_filter(transl_np[b, :, j], 11, 5)
        transl_s = torch.from_numpy(transl_np).to(latent.device)

        k3d = None
        if net.body_model:
            # Run body_model on CPU (its internal tensors are on CPU)
            rot6d_cpu = rot6d_s.cpu()
            transl_cpu = transl_s.cpu()
            k3d_list = []
            vertices_list = []
            with torch.no_grad():
                for b in range(B):
                    out = net.body_model.forward({"rot6d": rot6d_cpu[b], "trans": transl_cpu[b]})
                    k3d_list.append(out["keypoints3d"])
                    vertices_list.append(out["vertices"])
            k3d = torch.stack(k3d_list, dim=0)
            vertices = torch.stack(vertices_list, dim=0)
            # Align to ground
            min_y = vertices[..., 1].amin(dim=(1, 2), keepdim=True)
            k3d = k3d.clone()
            k3d[..., 1] -= min_y
            transl_cpu = transl_cpu.clone()
            transl_cpu[..., 1] -= min_y.squeeze(-1)
            transl_s = transl_cpu
        else:
            k3d = torch.zeros(B, L, 22, 3)

        return {"keypoints3d": k3d.cpu(), "rot6d": rot6d_s.cpu(), "transl": transl_s.cpu(),
                "root_rotations_mat": rot6d_to_rotation_matrix(rot6d_s[:, :, 0].cpu()).cpu()}


# ============================================================================
# Node 5: HYMotion Preview
# ============================================================================

class HYMotionPreview:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "motion_data": ("HYMOTION_DATA",),
            "sample_index": ("INT", {"default": 0, "min": 0, "max": 3}),
            "frame_step": ("INT", {"default": 5, "min": 1, "max": 30}),
            "image_size": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 64}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "render"
    CATEGORY = "HY-Motion"
    OUTPUT_NODE = True

    def render(self, motion_data, sample_index=0, frame_step=5, image_size=512):
        kpts = motion_data.output_dict.get("keypoints3d")
        if kpts is None:
            return (torch.zeros(1, image_size, image_size, 3),)

        kpts = kpts[min(sample_index, kpts.shape[0]-1)].cpu().numpy()
        frames = [self._draw(kpts[i], image_size) for i in range(0, len(kpts), frame_step)]
        return (torch.from_numpy(np.stack(frames)).float() / 255.0,)

    def _draw(self, kpts, size):
        img = np.ones((size, size, 3), dtype=np.uint8) * 240
        # Only use first 22 main joints for skeleton (ignore fingers)
        bones = [(0,1),(1,4),(4,7),(7,10),(0,2),(2,5),(5,8),(8,11),(0,3),(3,6),(6,9),(9,12),(12,15),(9,13),(13,16),(16,18),(18,20),(9,14),(14,17),(17,19),(19,21)]
        colors = [(255,100,100)]*4 + [(100,100,255)]*4 + [(100,200,100)]*5 + [(255,150,50)]*4 + [(50,150,255)]*4

        # Only take first 22 joints for drawing
        kpts_draw = kpts[:22] if len(kpts) > 22 else kpts
        x, y = kpts_draw[:, 0], kpts_draw[:, 1]
        cx, cy = (x.min()+x.max())/2, (y.min()+y.max())/2
        scale = max(x.max()-x.min(), y.max()-y.min(), 0.1) * 1.3

        def px(p): return int((p[0]-cx)/scale*size + size/2), int(size/2 - (p[1]-cy)/scale*size)

        for (a, b), c in zip(bones, colors):
            if a < len(kpts_draw) and b < len(kpts_draw):
                p1, p2 = px(kpts_draw[a]), px(kpts_draw[b])
                steps = max(abs(p2[0]-p1[0]), abs(p2[1]-p1[1]), 1) + 1
                for t in np.linspace(0, 1, int(steps)):
                    px_, py_ = int(p1[0]+t*(p2[0]-p1[0])), int(p1[1]+t*(p2[1]-p1[1]))
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            if 0 <= px_+dx < size and 0 <= py_+dy < size:
                                img[py_+dy, px_+dx] = c

        for i in range(len(kpts_draw)):
            p = px(kpts_draw[i])
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    if dx*dx+dy*dy <= 16 and 0 <= p[0]+dx < size and 0 <= p[1]+dy < size:
                        img[p[1]+dy, p[0]+dx] = [50, 50, 50]
        return img


# ============================================================================
# Node 6: HYMotion Save NPZ
# ============================================================================

class HYMotionSaveNPZ:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "motion_data": ("HYMOTION_DATA",),
            "output_dir": ("STRING", {"default": "hymotion_npz"}),
            "filename_prefix": ("STRING", {"default": "motion"}),
        }}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "save"
    CATEGORY = "HY-Motion"
    OUTPUT_NODE = True

    def save(self, motion_data, output_dir, filename_prefix):
        out_dir = os.path.join(COMFY_OUTPUT_DIR, output_dir)
        os.makedirs(out_dir, exist_ok=True)
        ts, uid = get_timestamp(), str(uuid.uuid4())[:8]

        paths = []
        for i in range(motion_data.batch_size):
            data = {k: motion_data.output_dict[k][i].cpu().numpy() if hasattr(motion_data.output_dict[k][i], 'cpu') else motion_data.output_dict[k][i]
                    for k in ["keypoints3d", "rot6d", "transl", "root_rotations_mat"] if k in motion_data.output_dict}
            data.update({"text": motion_data.text, "duration": motion_data.duration, "seed": motion_data.seeds[i] if i < len(motion_data.seeds) else 0})
            path = os.path.join(out_dir, f"{filename_prefix}_{ts}_{uid}_{i:03d}.npz")
            np.savez(path, **data)
            paths.append(path)
            print(f"[HY-Motion] Saved: {path}")

        return ("\n".join([os.path.relpath(p, COMFY_OUTPUT_DIR) for p in paths]),)


# ============================================================================
# Node 7: HYMotion Export FBX
# ============================================================================

class HYMotionExportFBX:
    _fbx_converter = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "motion_data": ("HYMOTION_DATA",),
            "output_dir": ("STRING", {"default": "hymotion_fbx"}),
            "filename_prefix": ("STRING", {"default": "motion"}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("fbx_paths",)
    FUNCTION = "export"
    CATEGORY = "HY-Motion"
    OUTPUT_NODE = True

    def export(self, motion_data, output_dir, filename_prefix):
        from .hymotion.pipeline.body_model import construct_smpl_data_dict

        # Lazy load FBX converter
        if HYMotionExportFBX._fbx_converter is None:
            try:
                import fbx
                from .hymotion.utils.smplh2woodfbx import SMPLH2WoodFBX
                template_path = os.path.join(CURRENT_DIR, "assets", "wooden_models", "boy_Rigging_smplx_tex.fbx")
                HYMotionExportFBX._fbx_converter = SMPLH2WoodFBX(template_fbx_path=template_path)
                print("[HY-Motion] FBX converter loaded")
            except ImportError:
                return ("FBX SDK not installed",)
            except Exception as e:
                return (f"FBX converter error: {e}",)

        out_dir = os.path.join(COMFY_OUTPUT_DIR, output_dir)
        os.makedirs(out_dir, exist_ok=True)
        ts, uid = get_timestamp(), str(uuid.uuid4())[:8]

        paths = []
        for i in range(motion_data.batch_size):
            try:
                rot6d = motion_data.output_dict["rot6d"][i].clone()
                transl = motion_data.output_dict["transl"][i].clone()
                smpl_data = construct_smpl_data_dict(rot6d, transl)

                path = os.path.join(out_dir, f"{filename_prefix}_{ts}_{uid}_{i:03d}.fbx")
                success = HYMotionExportFBX._fbx_converter.convert_npz_to_fbx(smpl_data, path)

                if success:
                    paths.append(path)
                    print(f"[HY-Motion] FBX saved: {path}")
                    # Save text description
                    txt_path = path.replace(".fbx", ".txt")
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(motion_data.text)
            except Exception as e:
                print(f"[HY-Motion] FBX export error: {e}")

        if not paths:
            return ("Export failed",)
        return ("\n".join([os.path.relpath(p, COMFY_OUTPUT_DIR) for p in paths]),)


# ============================================================================
# Node 8: HYMotion Preview Animation (Three.js with GLB Export)
# ============================================================================

class HYMotionPreviewAnimation:
    """
    Interactive 3D motion preview with Three.js.
    Supports playback controls and GLB export with skeleton animation.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion_data": ("HYMOTION_DATA",),
            },
            "optional": {
                "sample_index": ("INT", {"default": 0, "min": 0, "max": 3}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("motion_data",)
    FUNCTION = "preview"
    CATEGORY = "HY-Motion"
    OUTPUT_NODE = True

    def preview(self, motion_data: HYMotionData, sample_index: int = 0):
        import json
        from .hymotion.utils.geometry import rot6d_to_rotation_matrix, matrix_to_quaternion

        idx = min(sample_index, motion_data.batch_size - 1)

        # Extract motion data for the selected sample
        rot6d = motion_data.output_dict["rot6d"][idx]  # (num_frames, 22, 6)
        transl = motion_data.output_dict["transl"][idx]  # (num_frames, 3)

        # Convert rot6d to quaternion using the same math as FBX export
        # rot6d -> rotation_matrix -> quaternion
        if hasattr(rot6d, 'cpu'):
            rot6d_tensor = rot6d.cpu()
        else:
            rot6d_tensor = torch.from_numpy(rot6d).float()

        # (num_frames, 22, 6) -> (num_frames, 22, 3, 3)
        rot_matrices = rot6d_to_rotation_matrix(rot6d_tensor)
        # (num_frames, 22, 3, 3) -> (num_frames, 22, 4) quaternion [w, x, y, z]
        quaternions = matrix_to_quaternion(rot_matrices)

        # Convert to numpy
        quaternions_np = quaternions.numpy()
        if hasattr(transl, 'cpu'):
            transl_np = transl.cpu().numpy()
        else:
            transl_np = transl

        num_frames = quaternions_np.shape[0]
        num_joints = quaternions_np.shape[1]

        # Flatten for transfer: quaternions are [w, x, y, z] format
        motion_json = json.dumps({
            "quaternions": quaternions_np.flatten().tolist(),  # (num_frames * num_joints * 4)
            "transl": transl_np.flatten().tolist(),  # (num_frames * 3)
            "num_frames": int(num_frames),
            "num_joints": int(num_joints),
            "fps": 30,
            "text": motion_data.text,
            "duration": motion_data.duration,
        })

        print(f"[HY-Motion] Preview Animation: {num_frames} frames, {num_joints} joints")
        return {"ui": {"motion_data": [motion_json]}, "result": (motion_json,)}


# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "HYMotionLoadLLM": HYMotionLoadLLM,
    "HYMotionLoadLLMGGUF": HYMotionLoadLLMGGUF,
    "HYMotionLoadNetwork": HYMotionLoadNetwork,
    "HYMotionLoadPrompter": HYMotionLoadPrompter,
    "HYMotionRewritePrompt": HYMotionRewritePrompt,
    "HYMotionEncodeText": HYMotionEncodeText,
    "HYMotionGenerate": HYMotionGenerate,
    "HYMotionPreview": HYMotionPreview,
    "HYMotionSaveNPZ": HYMotionSaveNPZ,
    "HYMotionExportFBX": HYMotionExportFBX,
    "HYMotionPreviewAnimation": HYMotionPreviewAnimation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HYMotionLoadLLM": "HY-Motion Load LLM",
    "HYMotionLoadLLMGGUF": "HY-Motion Load LLM (GGUF)",
    "HYMotionLoadNetwork": "HY-Motion Load Network",
    "HYMotionLoadPrompter": "HY-Motion Load Prompter",
    "HYMotionRewritePrompt": "HY-Motion Rewrite Prompt",
    "HYMotionEncodeText": "HY-Motion Encode Text",
    "HYMotionGenerate": "HY-Motion Generate",
    "HYMotionPreview": "HY-Motion Preview",
    "HYMotionSaveNPZ": "HY-Motion Save NPZ",
    "HYMotionExportFBX": "HY-Motion Export FBX",
    "HYMotionPreviewAnimation": "HY-Motion Preview Animation (3D)",
}

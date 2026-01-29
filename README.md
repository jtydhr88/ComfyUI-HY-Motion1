# ComfyUI-HY-Motion1

此版本扩展由 b站 点满兴趣的冒险者 修改。792818521@qq.com

开放了大模型选择范围，修复了显存溢出等问题。完善了部署文档。

A ComfyUI plugin based on [HY-Motion 1.0](https://github.com/Tencent-Hunyuan/HY-Motion-1.0) for text-to-3D human motion generation.

## Features

- **Text-to-Motion Generation**: Generate 3D human motion from text descriptions
- **Prompt Rewrite**: Automatically optimize text prompts and estimate motion duration using LLM
- **Multi-sample Generation**: Generate multiple motion samples simultaneously
- **Motion Preview**: Real-time skeleton preview rendering
- **3D Animation Preview**: Interactive Three.js viewer with playback controls
- **GLB Export**: Export to GLB format with skeleton animation (no dependencies required)
- **FBX Export**: Export to standard FBX format for Maya/Blender and other DCC tools
- **Custom Character Retargeting**: Retarget motion to custom FBX models (Mixamo supported, right now)
- **NPZ Save**: Save in universal NPZ format
- **GGUF Support**: Load quantized Qwen3-8B GGUF models for lower VRAM usage
- **CPU Offload**: Option to run LLM models on CPU to save GPU VRAM

## Installation

### 1. Install Dependencies

```bash
cd ComfyUI/custom_nodes/ComfyUI-HY-Motion1
pip install -r requirements.txt
```

### 2. Download Model Weights

Place model weights in ComfyUI's models directory:

```
ComfyUI/
└── models/
    └── HY-Motion/
        └── ckpts/
            ├── tencent/
            │   ├── HY-Motion-1.0/
            │   │   ├── config.yml
            │   │   └── latest.ckpt
            │   └── HY-Motion-1.0-Lite/
            │       ├── config.yml
            │       └── latest.ckpt
            ├── GGUF/                    # Optional: for GGUF models
            │   └── Qwen3-8B-Q4_K_M.gguf
            ├── clip-vit-large-patch14   # Required for text encoding
            ├── Qwen3-0.6B              # Optional: lightweight alternative
            ├── Qwen3-8B-bnb-4bit/       # Optional: bnb-4bit quantized
            └── Qwen3-8B-AWQ/           # Optional: AWQ quantized
```

### 3. Install Additional Dependencies (Optional)

#### For AWQ Models

To use AWQ quantized models, install the auto-awq package:

```bash
pip install auto-awq
```

If auto-awq installation fails, the system will automatically fall back to regular loading.

### 4. Model Download Links

| Model             | Source                                                                    | VRAM Required |
| ----------------- | ------------------------------------------------------------------------- | ------------- |
| Qwen3-8B-bnb-4bit | [unsloth/Qwen3-8B-bnb-4bit](https://hf-mirror.com/unsloth/Qwen3-8B-bnb-4bit) | ~4-5GB        |
| Qwen3-8B-AWQ      | [Qwen/Qwen3-8B-AWQ](https://hf-mirror.com/Qwen/Qwen3-8B-AWQ)                 | ~3-4GB        |
| Qwen3-8B-GGUF     | [Qwen/Qwen3-8B-GGUF](https://hf-mirror.com/Qwen/Qwen3-8B-GGUF)               | ~5GB (Q4_K_M) |
| HY-Motion-1.0     | [tencent/HY-Motion-1.0](https://hf-mirror.com/tencent/HY-Motion-1.0)         | ~8GB+         |

Download using huggingface-cli:

```bash
# Create directory first
mkdir -p models/HY-Motion/ckpts/tencent

# Download models
huggingface-cli download tencent/HY-Motion-1.0 --local-dir models/HY-Motion/ckpts/tencent
```

or manually download from https://huggingface.co/tencent/HY-Motion-1.0/tree/main

## Node Documentation

### HY-Motion Load LLM

Load Qwen3-8B LLM from HuggingFace (supports BitsAndBytes quantization).

| Parameter      | Description                                                               |
| -------------- | ------------------------------------------------------------------------- |
| model_name     | Select Qwen3 model from the list                                          |
| quantization   | Quantization mode:`none` / `int8` / `int4` / `bnb-4bit` / `awq` |
| offload_to_cpu | Load model on CPU instead of GPU (slower but saves VRAM)                  |

### Supported Models

#### Qwen3-8B-bnb-4bit

- **Source**: [unsloth/Qwen3-8B-bnb-4bit](https://hf-mirror.com/unsloth/Qwen3-8B-bnb-4bit)
- **Quantization**: BitsAndBytes 4-bit
- **VRAM Required**: ~4-5GB
- **Installation**: Place in `ComfyUI/models/HY-Motion/ckpts/Qwen3-8B-bnb-4bit/`

#### Qwen3-8B-AWQ

- **Source**: [Qwen/Qwen3-8B-AWQ](https://hf-mirror.com/Qwen/Qwen3-8B-AWQ)
- **Quantization**: AWQ (Activation-aware Weight Quantization)
- **VRAM Required**: ~3-4GB
- **Installation**: Place in `ComfyUI/models/HY-Motion/ckpts/Qwen3-8B-AWQ/`

#### Traditional Models

- **Qwen3-8B**: Full precision (16GB+ VRAM)
- **Qwen3-8B-int8**: 8-bit quantization (~8GB VRAM)
- **Qwen3-8B-int4**: 4-bit quantization (~4GB VRAM)

### Memory Optimization

The new quantized models offer significant memory savings:

| Model Format      | VRAM Required | Loading Speed | Quality    |
| ----------------- | ------------- | ------------- | ---------- |
| Qwen3-8B (full)   | ~16GB         | Slow          | Best       |
| Qwen3-8B-int8     | ~8GB          | Medium        | Good       |
| Qwen3-8B-int4     | ~4GB          | Fast          | Acceptable |
| Qwen3-8B-bnb-4bit | ~4-5GB        | Fast          | Good       |
| Qwen3-8B-AWQ      | ~3-4GB        | Very Fast     | Very Good  |

**Recommendation**: For most systems, use Qwen3-8B-AWQ or Qwen3-8B-bnb-4bit for the best balance of speed and quality with minimal memory usage.

### HY-Motion Load LLM (GGUF)

Load Qwen3-8B LLM from GGUF file.

| Parameter      | Description                                              |
| -------------- | -------------------------------------------------------- |
| gguf_file      | Select GGUF file from the list                           |
| offload_to_cpu | Load model on CPU instead of GPU (slower but saves VRAM) |

### HY-Motion Load Prompter

Load Text2MotionPrompter LLM for prompt rewriting and duration estimation.

| Parameter      | Description                                              |
| -------------- | -------------------------------------------------------- |
| model_source   | Model source:`(auto download)` or local path           |
| offload_to_cpu | Load model on CPU instead of GPU (slower but saves VRAM) |

**Note**: The model will be automatically downloaded from HuggingFace on first use (~2-3GB).

### HY-Motion Rewrite Prompt

Rewrite text prompt and estimate motion duration using LLM.

| Parameter | Description                                              |
| --------- | -------------------------------------------------------- |
| prompter  | Prompter model from Load Prompter node                   |
| text      | Original text description (supports Chinese and English) |

| Output         | Description                          |
| -------------- | ------------------------------------ |
| rewritten_text | Optimized English description        |
| duration       | Estimated motion duration in seconds |

**Note**: You need to download GGUF files manually from https://huggingface.co/Qwen/Qwen3-8B-GGUF

Place GGUF files in: `ComfyUI/models/HY-Motion/ckpts/GGUF/`

Recommended GGUF versions:

| File                 | Size    | Description                                    |
| -------------------- | ------- | ---------------------------------------------- |
| Qwen3-8B-Q4_K_M.gguf | 5.03 GB | Best balance of quality and size (recommended) |
| Qwen3-8B-Q5_K_M.gguf | 5.85 GB | Higher quality                                 |
| Qwen3-8B-Q6_K.gguf   | 6.73 GB | Near original quality                          |
| Qwen3-8B-Q8_0.gguf   | ~8 GB   | Almost lossless                                |

### HY-Motion Load Network

Load Motion Diffusion Network.

| Parameter  | Description                                                      |
| ---------- | ---------------------------------------------------------------- |
| model_name | Select model version:`HY-Motion-1.0` or `HY-Motion-1.0-Lite` |

### HY-Motion Encode Text

Encode text prompt for motion generation.

| Parameter | Description                  |
| --------- | ---------------------------- |
| llm       | LLM model from Load LLM node |
| text      | Motion description text      |

### HY-Motion Generate

Core generation node.

| Parameter    | Description                        |
| ------------ | ---------------------------------- |
| network      | Network from Load Network node     |
| conditioning | Conditioning from Encode Text node |
| duration     | Motion duration (seconds)          |
| seed         | Random seed                        |
| cfg_scale    | Text guidance scale                |
| num_samples  | Number of samples to generate      |

### HY-Motion Preview

Render skeleton preview images (2D frame sequence).

### HY-Motion Preview Animation (3D)

Interactive 3D animation preview with Three.js viewer.

| Feature  | Description                                   |
| -------- | --------------------------------------------- |
| Playback | Play/pause, speed control, timeline scrubbing |
| Display  | Toggle skeleton, mesh, grid visibility        |
| Export   | Download GLB file with skeleton animation     |

**This node provides a pure frontend GLB export that requires no additional Python dependencies.**

> **Note**: This node does NOT automatically save files. You must manually click the "Export GLB" button in the viewer to download the animation file.

### HY-Motion Export FBX

Export FBX file with optional custom Mixamo character retargeting (requires fbxsdkpy installation).

| Parameter       | Description                                                |
| --------------- | ---------------------------------------------------------- |
| motion_data     | Motion data from Generate node                             |
| output_dir      | Output subdirectory in ComfyUI output folder               |
| filename_prefix | Prefix for output filenames                                |
| custom_fbx_path | (Optional) Path to custom FBX model for retargeting        |
| yaw_offset      | (Optional) Y-axis rotation offset in degrees (-180 to 180) |
| scale           | (Optional) Force scale multiplier (0 = auto)               |

#### Custom FBX Path Rules

| Input                  | Resolved Path                                     |
| ---------------------- | ------------------------------------------------- |
| `3d/char.fbx`        | `ComfyUI/input/3d/char.fbx` (default to input/) |
| `input/3d/char.fbx`  | `ComfyUI/input/3d/char.fbx`                     |
| `output/3d/char.fbx` | `ComfyUI/output/3d/char.fbx`                    |
| `D:\Models\char.fbx` | `D:\Models\char.fbx` (absolute path)            |
| (empty)                | Uses default wooden boy model                     |

#### Supported Rigs

- **Mixamo**: Full automatic bone mapping with `mixamorig:` prefix

> **Note**: The retargeting code (`retarget_fbx.py`) is adapted from [ComfyUI-HyMotion](https://github.com/Aero-Ex/ComfyUI-HyMotion).

### HY-Motion Save NPZ

Save in NPZ format.

## Example Workflow

### Basic Workflow

```
[HY-Motion Load LLM] ──┐
                       ├──> [HY-Motion Encode Text] ──┐
[HY-Motion Load Network] ─────────────────────────────┴──> [HY-Motion Generate] ──┬──> [HY-Motion Preview]
                                                                                  ├──> [HY-Motion Preview Animation (3D)] ──> Export GLB
                                                                                  ├──> [HY-Motion Save NPZ]
                                                                                  └──> [HY-Motion Export FBX]
```

### With Prompt Rewrite (Recommended)

```
[HY-Motion Load Prompter] ──> [HY-Motion Rewrite Prompt] ──┬──> rewritten_text ──> [HY-Motion Encode Text]
                                      │                    │
                                      │                    └──> duration ──> [HY-Motion Generate]
                                      │
                                      └── text (user input)

[HY-Motion Load LLM] ──────────────────────────────────────────> [HY-Motion Encode Text] ──┐
                                                                                           │
[HY-Motion Load Network] ─────────────────────────────────────────────────────────────────┴──> [HY-Motion Generate]
```

The Prompt Rewrite workflow:

1. Takes your text input (supports Chinese/English)
2. Optimizes it to a standardized English description
3. Estimates appropriate motion duration
4. Feeds both to the generation pipeline

### For GGUF

```
[HY-Motion Load LLM (GGUF)] ──> [HY-Motion Encode Text] ──> ...
```

## Notes

1. **VRAM Requirements**:

   - HY-Motion-1.0: ~8GB+ VRAM (model only)
   - HY-Motion-1.0-Lite: ~4GB+ VRAM (model only)
   - Qwen3-8B Text Encoder (additional):
     - HuggingFace `quantization=none`: ~16GB VRAM
     - HuggingFace `quantization=int8`: ~8GB VRAM
     - HuggingFace `quantization=int4`: ~4GB VRAM
     - GGUF Q4_K_M: ~5GB VRAM
   - Text2MotionPrompter (optional): ~2-3GB VRAM (4bit quantized)
2. **CPU Offload**:

   - All LLM loader nodes support `offload_to_cpu` option
   - When enabled, the model runs entirely on CPU (no GPU VRAM required)
   - Trade-off: Slower inference speed but allows running multiple LLMs simultaneously
   - Recommended: Enable CPU offload for Prompter if VRAM is limited, keep Text Encoder on GPU for faster encoding
3. **GGUF Requirements**:

   - Requires `transformers>=4.40`
   - GGUF files must be downloaded manually
   - Place in `ComfyUI/models/HY-Motion/ckpts/GGUF/`
4. **FBX Export**: Requires additional fbxsdkpy installation:

   ```bash
   pip install fbxsdkpy --extra-index-url https://gitlab.inria.fr/api/v4/projects/18692/packages/pypi/simple
   ```

   **Having trouble installing fbxsdkpy?** Use the **HY-Motion Preview Animation (3D)** node instead! It provides a pure frontend GLB export with skeleton animation that works without any additional Python dependencies.
5. **Text Encoder**: CLIP model will be downloaded automatically on first use. Qwen3-8B will be downloaded automatically when using Load LLM node (not GGUF).
6. **Prompt Rewrite**: Text2MotionPrompter model will be downloaded automatically on first use (~2-3GB). Supports Chinese and English input.

## License

Please refer to the HY-Motion 1.0 original project license.

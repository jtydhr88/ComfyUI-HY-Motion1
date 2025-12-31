# ComfyUI-HY-Motion1

A ComfyUI plugin based on [HY-Motion 1.0](https://github.com/Tencent-Hunyuan/HY-Motion-1.0) for text-to-3D human motion generation.

## Features

- **Text-to-Motion Generation**: Generate 3D human motion from text descriptions
- **Multi-sample Generation**: Generate multiple motion samples simultaneously
- **Motion Preview**: Real-time skeleton preview rendering
- **FBX Export**: Export to standard FBX format for Maya/Blender and other DCC tools
- **NPZ Save**: Save in universal NPZ format

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
            └── tencent/
                ├── HY-Motion-1.0/
                │   ├── config.yml
                │   └── latest.ckpt
                └── HY-Motion-1.0-Lite/
                    ├── config.yml
                    └── latest.ckpt
```

Download using huggingface-cli:

```bash
# Create directory first
mkdir -p models/HY-Motion/ckpts/tencent

# Download models
huggingface-cli download tencent/HY-Motion-1.0 --local-dir models/HY-Motion/ckpts/tencent
```

or manually download from https://huggingface.co/tencent/HY-Motion-1.0/tree/main

## Node Documentation

### HY-Motion Load Model
Load HY-Motion model.

| Parameter | Description |
|-----------|-------------|
| model_name | Select model version (1.0B or Lite 0.46B) |
| device | Runtime device (cuda/cpu) |

### HY-Motion Generate
Core generation node.

| Parameter | Description |
|-----------|-------------|
| text | Motion description text |
| duration | Motion duration (seconds) |
| seed | Random seed |
| cfg_scale | Text guidance scale |
| num_samples | Number of samples to generate |

### HY-Motion Preview
Render skeleton preview images.

### HY-Motion Export FBX
Export FBX file (requires fbxsdkpy installation).

### HY-Motion Save NPZ
Save in NPZ format.

## Example Workflow

Basic workflow:
```
HY-Motion Load Model -> HY-Motion Generate -> HY-Motion Preview
                                           -> HY-Motion Save NPZ
```

## Notes

1. **VRAM Requirements**:
   - HY-Motion-1.0: ~8GB+ VRAM
   - HY-Motion-1.0-Lite: ~4GB+ VRAM

2. **FBX Export**: Requires additional fbxsdkpy installation:
   ```bash
   pip install fbxsdkpy --extra-index-url https://gitlab.inria.fr/api/v4/projects/18692/packages/pypi/simple
   ```

3. **Text Encoder**: CLIP and Qwen3 models will be downloaded automatically on first use

## License

Please refer to the HY-Motion 1.0 original project license.

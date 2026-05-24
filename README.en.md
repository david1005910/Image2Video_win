# Image2Video_win

[한국어](README.md) | **English**

An **image→video generation pipeline** that turns a single text prompt into a short video clip.
It chains free/cheap services and local models, running on a GPU. This is a local port targeting a
**WSL2 + conda + RTX 3060** setup, and it also includes a web front end (`webapp/`) wrapping the same pipeline.

```
Pollinations (image) → Gemini (motion prompt, optional) → Wan 2.2 I2V (video)
  → RIFE / FFmpeg (frame interpolation) → Real-ESRGAN (upscale, optional) → FFmpeg (color grade / audio)
```

## Layout

| File | Description |
|------|-------------|
| `image_to_video_pipeline.py` | **The maintained artifact.** Local (WSL) CLI pipeline. |
| `image_to_video_pipeline.ipynb` | Original Google Colab notebook (T4 GPU). Reference only — additionally documents a ComfyUI path and a Veo 3.1 hybrid path. |
| `webapp/` | Full-stack web front end (FastAPI + vanilla JS) using the same pipeline. See [`webapp/README.md`](webapp/README.md). |

## Pipeline stages

`main()` runs six stages sequentially; each writes its output into a numbered subdirectory of the work folder (`01_images` … `06_final`).

1. **Image** — generate a PNG from the free Pollinations API (no API key needed).
2. **Motion prompt** — auto-generated via Gemini Vision if `GEMINI_API_KEY` is set, otherwise a default. (Can also be specified manually.)
3. **Video** — Wan 2.2 I2V (diffusers `WanImageToVideoPipeline`, bf16, CPU offload). The core of the pipeline.
4. **Frame interpolation** — Practical-RIFE (when weights are present) or FFmpeg `minterpolate` fallback.
5. **Upscale** — Real-ESRGAN (optional). Off by default since Vulkan is unreliable under WSL.
6. **Finalize** — FFmpeg color grade + fade-in, optional BGM mux → `06_final/final.mp4`.

## Requirements

- **WSL2** + a conda env named `img2vid` (CUDA 12.8 build of PyTorch)
- A **CUDA GPU** is required — the run aborts if no GPU is found.
- On PATH: `ffmpeg` (required), plus `git`/`wget`/`unzip`/`chmod` (when using RIFE / Real-ESRGAN)

## Install (one-time)

```bash
conda activate img2vid
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install "diffusers>=0.31" transformers accelerate safetensors ftfy \
            imageio imageio-ffmpeg requests pillow
pip install google-genai          # (optional) for Gemini motion-prompt generation
pip install gguf                   # (optional) when using GGUF quantization
sudo apt update && sudo apt install -y ffmpeg
hf auth login                      # HuggingFace auth for model downloads
```

## Run

```bash
conda activate img2vid
python image_to_video_pipeline.py

# Flags that override CONFIG values:
python image_to_video_pipeline.py --prompt "a neon city at night" --aspect 9:16 --steps 30 --upscale
python image_to_video_pipeline.py --gguf Q5_K_M    # GGUF-quantized transformer (big VRAM cut)
```

Available flags: `--prompt` `--motion` `--aspect {16:9,9:16,1:1}` `--width` `--steps` `--upscale` `--gguf`

Quick GPU-free check (arg parsing / import only):

```bash
python image_to_video_pipeline.py --help
```

## Configuration

All tunables live in the `CONFIG` dict near the top of the `.py`. CLI flags override only certain keys; everything else is edited directly in `CONFIG`.

- **`base_width`** — the main VRAM lever. 832–1024 recommended on a 12 GB 3060 (1280 risks OOM).
- **`gguf_quant`** — GGUF-quantized transformer (e.g. `Q5_K_M`). Big VRAM cut: bf16 ~10 GB → Q5 ~3.8 GB.
- **`model_id`** — swap lightweight `Wan2.2-TI2V-5B` ↔ higher-quality `Wan2.2-I2V-A14B`.

### Environment variables

| Variable | Description |
|----------|-------------|
| `IMG2VID_WORK` | Work directory (default `/home/sharkey/img2video_work`). **All outputs land here, outside the repo.** |
| `HF_HOME` | HuggingFace model cache (default `/home/sharkey/hf_cache`). |
| `GEMINI_API_KEY` | (optional) Enables auto motion-prompt generation. Never hardcode it. |
| `HF_TOKEN` | Auto-injected from the cached token if unset. |

> **WSL download stabilization:** at import time it forces IPv4, disables the Xet backend, and injects `HF_TOKEN` (working around WSL2's IPv6 stalling large HF downloads). This is a deliberate workaround.

## Web app

A buildless web front end that reuses the CLI pipeline unchanged.

```bash
conda activate img2vid
pip install -r webapp/requirements.txt   # one-time (fastapi / uvicorn / python-multipart)
./webapp/run.sh                          # → http://localhost:8000
```

With a single GPU, jobs run serially; stage progress is shown in real time and the finished video is embedded on completion.
See [`webapp/README.md`](webapp/README.md) for details.

## Notes

- Comments, docstrings, and console output in the code are written in **Korean**.
- There is no test suite or build system. Validate changes with a real run or the GPU-free dry-check above.
- The notebook (`.ipynb`) is reference only and additionally covers the ComfyUI / Veo 3.1 hybrid paths that the `.py` omits.

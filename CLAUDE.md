# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A GPU-driven image→video generation pipeline. It chains free/cheap services and local models to turn a text prompt into a finished short video clip. The core is one runnable CLI script (plus the notebook it was ported from); there is no test suite and no build system. A buildless full-stack web front end wrapping the same pipeline lives under `webapp/` (see "Web app" below).

- `image_to_video_pipeline.py` — the maintained artifact. A local **WSL + conda + RTX 3060** port of the notebook.
- `image_to_video_pipeline.ipynb` — the original **Google Colab** version (T4 GPU, Drive mount). Keep as reference only; it additionally documents two alternatives the `.py` omits: a **ComfyUI** path and a **Veo 3.1 (paid Gemini API) hybrid** path. When asked about those, read the notebook cells.
- `webapp/` — a full-stack web front end (FastAPI + vanilla JS) for the pipeline.

Comments, docstrings, and console output are written in **Korean** — match that style when editing the `.py`.

## Runtime environment

- Runs under **WSL2** in a conda env named `img2vid` (Python, CUDA 12.8 build of PyTorch). Activate it before running: `conda activate img2vid`.
- **Do NOT use the in-repo `.venv`** — it is a Windows uv venv (its `pyvenv.cfg` `home` points at a Windows path) and is not usable from WSL. It is leftover, not the runtime.
- Requires a CUDA GPU; `check_gpu()` aborts the run if `torch.cuda.is_available()` is false.
- External CLI tools must be on PATH: `ffmpeg` (required), plus `git`/`wget`/`unzip`/`chmod` (only when RIFE or Real-ESRGAN are enabled).
- **Model downloads are stabilized for WSL at import time** by `_stabilize_downloads()` (duplicated in the `.py` and `webapp/settings.py`): it forces IPv4 (WSL2 IPv6 stalls large HF downloads), disables the Xet backend (`HF_HUB_DISABLE_XET=1`), sets `HF_HUB_DOWNLOAD_TIMEOUT`, and injects `HF_TOKEN` from the cache so an overridden `HF_HOME` doesn't silently drop to slow unauthenticated downloads. This is a deliberate workaround — keep it if you touch import-time setup.

## Running

```bash
conda activate img2vid
python image_to_video_pipeline.py
# overrides (each maps onto the CONFIG dict):
python image_to_video_pipeline.py --prompt "a neon city at night" --aspect 9:16 --steps 30 --upscale
# available flags: --prompt --motion --aspect {16:9,9:16,1:1} --width --steps --upscale --gguf
python image_to_video_pipeline.py --gguf Q5_K_M   # GGUF-quantized transformer (big VRAM cut)
```

There is no lint/test target. To validate a change, do a real run (it is slow — minutes on a 3060) or dry-check without a GPU: `python image_to_video_pipeline.py --help` (and `python -c "import image_to_video_pipeline"`) parse args and import the module without touching torch/CUDA, since the heavy imports are lazy inside the `step_*` functions.

## Configuration model

All tunables live in the `CONFIG` dict near the top of the `.py`. CLI flags in `parse_args()`/`main()` override specific keys; everything else is edited in `CONFIG` directly. When adding a knob, add it to `CONFIG` and (if it should be runtime-settable) wire a matching flag in both `parse_args()` and the override block of `main()`.

**GGUF quantization** is the heavyweight VRAM lever, separate from `base_width`. Setting `gguf_quant` (e.g. `Q5_K_M`, default `None` = plain bf16) makes `load_wan_pipeline()` call `_load_gguf_transformer()` to load a quantized transformer (~bf16 10 GB → Q5 ~3.8 GB). Quirks worth knowing: `from_single_file` mis-detects the config as 14B, so the model's own `transformer` config dir is passed explicitly; `gguf_dir` is checked first so a pre-downloaded GGUF isn't re-fetched, otherwise it pulls from `gguf_repo`. Requires the `gguf` package.

Environment variables that matter:
- `IMG2VID_WORK` — work directory (default `/home/sharkey/img2video_work`). **All outputs land here, outside the repo.**
- `HF_HOME` — HuggingFace model cache (defaults to `/home/sharkey/hf_cache`); set via `os.environ.setdefault`.
- `GEMINI_API_KEY` — optional; enables auto motion-prompt generation. Never hardcode it.
- `HF_TOKEN` — auto-injected by `_stabilize_downloads()` from the cached token if unset; also honored if you export it. (`HF_HUB_DISABLE_XET`/`HF_HUB_DOWNLOAD_TIMEOUT` are set there too.)

## Pipeline architecture

`main()` orchestrates six sequential stages, each a `step_*` function that takes `CONFIG` plus the previous stage's output path and returns the next path. Stages write into numbered subdirectories of `IMG2VID_WORK` (`01_images` … `06_final`), created by `ensure_dirs()`, so the directory layout *is* the pipeline:

1. **Image** (`step_image`) — `generate_pollinations_image()` fetches a PNG from the free Pollinations HTTP API (no key; retries 3×).
2. **Motion prompt** (`step_motion_prompt`) — resolution order: explicit `CONFIG["motion_prompt"]` → Gemini Vision (`gemini-2.5-flash`, only if `GEMINI_API_KEY` set) → `DEFAULT_MOTION` fallback. Never hard-fails.
3. **Video** (`load_wan_pipeline` + `step_video`) — the core. Wan 2.2 I2V via diffusers `WanImageToVideoPipeline`, `bfloat16`, with `enable_model_cpu_offload()` for low VRAM. After this stage `main()` deletes the pipe and calls `torch.cuda.empty_cache()` before the ffmpeg/upscale stages — preserve that teardown if you reorder stages.
4. **Interpolate** (`step_interpolate`) — Practical-RIFE if `use_rife` *and* weights present, else **falls back to FFmpeg `minterpolate`**. RIFE is auto-cloned but its `train_log` weights must be downloaded manually.
5. **Upscale** (`step_upscale`) — optional Real-ESRGAN ncnn-vulkan (auto-downloaded). Vulkan is unreliable under WSL; off by default and a no-op pass-through when disabled.
6. **Finalize** (`step_finalize`) — FFmpeg color grade (`eq` contrast/saturation/brightness + fade-in) and optional BGM mux; emits `06_final/final.mp4`.

## Web app (`webapp/`)

A full-stack web front end that **reuses the CLI pipeline unchanged** — it does not reimplement any stage. Buildless: a FastAPI backend serves a vanilla-JS front end (no Node/npm). See `webapp/README.md` for run instructions. Launch from the repo root (so `image_to_video_pipeline` is importable):

```bash
conda activate img2vid
pip install -r webapp/requirements.txt   # fastapi, uvicorn, python-multipart (one-time)
./webapp/run.sh                           # → http://localhost:8000
```

Backend (`webapp/backend/`):
- `settings.py` — `DEFAULT_CONFIG` (a copy of the CLI's `CONFIG`) and `normalize(raw)`, which clamps/validates a web request dict into a safe CONFIG (numeric ranges, `base_width` floored to ×16, choice whitelists `IMAGE_MODELS`/`ASPECTS`/`WAN_MODELS`/`GGUF_QUANTS`). **When you change a knob in the CLI's `CONFIG`, mirror it here** — the two configs and the choice lists / front-end form must stay in sync. Also defines `WEBJOBS = WORK/"webjobs"` (per-job output root) and shared `TOOLS = WORK/"tools"`.
- `pipeline_runner.py` — `run_job()` runs the CLI's `step_*` functions for one job by reassigning the pipeline module's globals `P.WORK`/`P.TOOLS` to per-job paths, then calling the stages in `main()`'s order (incl. the post-video VRAM teardown). Reports stage progress via a callback. Raises instead of `sys.exit` on missing GPU.
- `jobs.py` — `JobManager`: a **single** daemon worker thread draining a queue, so jobs run strictly serially (one GPU; concurrent Wan loads would OOM). Holds `Job` state (status, per-stage status, captured log) **in memory only** — restarting the server loses the job list, though output files survive. Redirects `sys.stdout/stderr` into the job log during a run (acceptable because it's single-worker / single-user).
- `app.py` — FastAPI: `GET /api/options`, `POST /api/jobs` (multipart: `config` JSON string + optional `bgm` file), `GET /api/jobs`, `GET /api/jobs/{id}`. Mounts `/files` → `WEBJOBS` (serves `source.png`, `final.mp4`) and `/` → the front end (mount order matters: `/` last).

Front end (`webapp/frontend/`, plain `index.html`/`style.css`/`app.js`): a config form populated from `/api/options`, submits a job, then polls `/api/jobs/{id}` every 2 s to render the stage progress bar + log tail and embed the finished video.

## Conventions to follow
- All subprocess/shell work goes through the `run()` helper (it replaces the notebook's `!cmd`); prefer list-form commands over strings.
- Resolution comes from `resolve_resolution()`, which maps `aspect` → (w, h) and **floors both to multiples of 16** (Wan I2V requirement — *not* 8) — keep that constraint for any new aspect ratios.
- Optional stages are designed to degrade gracefully (skip or fall back) rather than crash; preserve that when extending them.
- `base_width` is the main VRAM lever (832–1024 recommended on a 12 GB 3060). `model_id` swaps the lightweight `Wan2.2-TI2V-5B` for the higher-quality `Wan2.2-I2V-A14B`.

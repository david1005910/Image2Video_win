#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image -> Video Pipeline (로컬 WSL 버전)
================================================================
Pollinations(이미지) -> Gemini(모션 프롬프트, 선택) -> Wan 2.2 I2V(영상)
 -> RIFE/FFmpeg(프레임 보간) -> Real-ESRGAN(영상 업스케일, 선택) -> FFmpeg(색보정/오디오)

Colab 노트북을 로컬(WSL + conda 'img2vid' + RTX 3060)에서 돌아가도록 변환한 스크립트.

----------------------------------------------------------------
[1회 의존성 설치] — img2vid 환경에서:
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    pip install "diffusers>=0.31" transformers accelerate safetensors ftfy \
                imageio imageio-ffmpeg requests pillow
    pip install google-genai          # (선택) Gemini 모션 프롬프트 자동 생성용
    sudo apt update && sudo apt install -y ffmpeg
    hf auth login                     # HuggingFace 로그인 (모델 다운로드 인증)

[환경변수 — 선택]
    export GEMINI_API_KEY="새로_발급받은_키"     # 코드에 키를 적지 마세요
    export IMG2VID_WORK="/home/sharkey/img2video_work"   # 작업 폴더 변경 시

[실행]
    python image_to_video_pipeline.py
    python image_to_video_pipeline.py --prompt "a neon city at night" --aspect 9:16
----------------------------------------------------------------
"""

import os
import sys
import time
import shutil
import argparse
import subprocess
import urllib.parse
from pathlib import Path

# ================================================================
# 작업 디렉터리
# ================================================================
WORK = os.environ.get("IMG2VID_WORK", "/home/sharkey/img2video_work")
TOOLS = os.path.join(WORK, "tools")  # 외부 도구(RIFE, Real-ESRGAN) 설치 위치

# 모델 캐시 위치 (WSL 네이티브 경로 권장 — 빠름)
os.environ.setdefault("HF_HOME", "/home/sharkey/hf_cache")


def _stabilize_downloads():
    """모델 다운로드 안정화 (WSL 환경 대응).

    - WSL2에서 IPv6 경로가 죽어 있으면 huggingface_hub/requests가 큰 파일에서
      그대로 멈춘다(작은 파일은 통과). 파이썬이 IPv4만 쓰도록 강제해 회피한다.
    - Xet 백엔드보다 클래식 CDN이 이 회선에서 빠르고 재개(resume)가 잘 된다.
    - HF_HOME을 바꾸면 기본 위치의 토큰을 못 찾아 비인증(저속)으로 받으므로 직접 주입.
    """
    import socket
    if not getattr(socket, "_ipv4_forced", False):
        _orig = socket.getaddrinfo
        socket.getaddrinfo = lambda host, *a, **k: [
            r for r in _orig(host, *a, **k) if r[0] == socket.AF_INET
        ]
        socket._ipv4_forced = True
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    if not os.environ.get("HF_TOKEN"):
        for p in (os.path.join(os.environ["HF_HOME"], "token"),
                  os.path.expanduser("~/.cache/huggingface/token")):
            if os.path.exists(p):
                os.environ["HF_TOKEN"] = open(p).read().strip()
                break


_stabilize_downloads()

# ================================================================
# 사용자 설정 (여기만 바꾸면 됩니다)
# ================================================================
CONFIG = {
    # --- 이미지 생성 ---
    "image_prompt": ("a serene mountain lake at golden hour, cinematic, "
                     "photorealistic, dramatic clouds reflecting on water"),
    "image_model": "flux",        # flux | turbo | flux-realism
    "image_seed": 42,             # 재현성 (None이면 랜덤)

    # --- 출력 포맷 ---
    "aspect": "16:9",             # "16:9" | "9:16"(숏폼) | "1:1"
    "base_width": 960,            # 영상 생성 기준. RTX 3060(12GB)은 832~1024 권장 (1280은 OOM 위험)

    # --- 영상 생성 (Wan 2.2 I2V) ---
    "model_id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",  # 경량(12GB VRAM용). 고품질: Wan2.2-I2V-A14B-Diffusers
    # GGUF 양자화 transformer 사용 시 VRAM 대폭 절감 (bf16 ~10GB → Q5 ~3.8GB).
    # None이면 일반 bf16. 예: "Q5_K_M"(추천) | "Q6_K" | "Q8_0" | "Q4_K_M"
    "gguf_quant": None,
    "gguf_repo": "QuantStack/Wan2.2-TI2V-5B-GGUF",   # gguf_quant 사용 시 가중치 출처(HF)
    "gguf_dir": "/home/sharkey/models/gguf",          # 로컬 GGUF 우선 탐색 위치(있으면 재다운로드 안 함)
    "motion_prompt": None,        # None이면 Gemini 자동 생성(키 있을 때). 직접 쓰려면 영어 문자열
    "video_length_frames": 49,    # 약 3초 @ 16fps
    "video_fps": 16,              # Wan 2.2 native fps
    "video_steps": 30,            # 추론 스텝 (높을수록 품질↑ 시간↑)
    "guidance_scale": 5.0,

    # --- 후처리 ---
    "interpolate_to_fps": 48,     # 보간 목표 fps (16 -> 48 = 3배)
    "use_rife": False,            # True면 Practical-RIFE 사용, False면 FFmpeg minterpolate
    "upscale_video": False,       # Real-ESRGAN 영상 업스케일 (Vulkan 필요, WSL에선 불안정할 수 있음)
    "upscale_factor": 2,

    # --- 색보정 (FFmpeg eq 필터) ---
    "contrast": 1.08,
    "saturation": 1.12,
    "brightness": 0.02,

    # --- 오디오 (선택) ---
    "bgm_path": None,             # 예: f"{WORK}/audio/bgm.mp3", 없으면 무음
}


# ================================================================
# 유틸리티
# ================================================================
def run(cmd, check=True, quiet=False, cwd=None):
    """셸 명령 실행 (리스트 권장). 노트북의 '!명령'을 대체."""
    if isinstance(cmd, str):
        printable = cmd
    else:
        printable = " ".join(str(c) for c in cmd)
    if not quiet:
        print(f"  $ {printable}")
    result = subprocess.run(
        cmd,
        shell=isinstance(cmd, str),
        cwd=cwd,
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=subprocess.DEVNULL if quiet else None,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"명령 실패(코드 {result.returncode}): {printable}")
    return result.returncode


def resolve_resolution(cfg):
    """종횡비 -> (width, height), 16의 배수로 정렬.

    Wan I2V 파이프라인은 height/width가 '16의 배수'여야 한다(8 아님).
    """
    bw = cfg["base_width"]
    table = {
        "16:9": (bw, int(bw * 9 / 16)),
        "9:16": (int(bw * 9 / 16), bw),
        "1:1":  (bw, bw),
    }
    w, h = table[cfg["aspect"]]
    cfg["width"] = (w // 16) * 16
    cfg["height"] = (h // 16) * 16
    return cfg["width"], cfg["height"]


def ensure_dirs():
    subdirs = ["01_images", "02_upscaled", "03_videos_raw",
               "04_interpolated", "05_upscaled_video", "06_final", "models"]
    for d in subdirs:
        os.makedirs(os.path.join(WORK, d), exist_ok=True)
    os.makedirs(TOOLS, exist_ok=True)
    print("작업 디렉터리:", WORK)


# ================================================================
# 0. GPU 확인
# ================================================================
def check_gpu():
    print("\n[0] GPU 확인")
    run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
         "--format=csv"], check=False)
    import torch
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        print("VRAM (GB):", vram)
        return True
    print("⚠️ GPU를 찾지 못했습니다. img2vid 환경/드라이버를 확인하세요.")
    return False


# ================================================================
# 1. Pollinations 이미지 생성 (무료, API 키 불필요)
# ================================================================
def generate_pollinations_image(prompt, w, h, model="flux", seed=42):
    import requests
    enc = urllib.parse.quote(prompt)
    url = (f"https://image.pollinations.ai/prompt/{enc}"
           f"?width={w}&height={h}&model={model}&nologo=true&enhance=true")
    if seed is not None:
        url += f"&seed={seed}"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=180)
            if r.status_code == 200 and len(r.content) > 5000:
                return r.content
            print(f"  재시도 {attempt + 1} (status={r.status_code})")
        except Exception as e:
            print(f"  재시도 {attempt + 1} (error={e})")
        time.sleep(5)
    raise RuntimeError("Pollinations 이미지 생성 실패")


def step_image(cfg):
    print("\n[1] 이미지 생성 (Pollinations)")
    img_path = os.path.join(WORK, "01_images", "source.png")
    content = generate_pollinations_image(
        cfg["image_prompt"], cfg["width"], cfg["height"],
        cfg["image_model"], cfg["image_seed"],
    )
    Path(img_path).write_bytes(content)
    print("✅ 이미지 저장:", img_path)
    return img_path


# ================================================================
# 2. 모션 프롬프트 (Gemini Vision, 선택)
# ================================================================
DEFAULT_MOTION = ("slow cinematic camera push-in, gentle ripples on water, "
                  "soft drifting clouds")


def step_motion_prompt(cfg, img_path):
    print("\n[2] 모션 프롬프트")
    motion = cfg.get("motion_prompt")
    if motion:
        print("📝 사용자 지정 모션 프롬프트:", motion)
        return motion

    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=api_key)
            image_bytes = Path(img_path).read_bytes()
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    ("Describe the most natural, cinematic motion to animate this "
                     "still image as a short video. Focus on subtle, physically "
                     "plausible camera movement and subject motion. Output ONLY a "
                     "concise English motion prompt, no preamble."),
                ],
                config=types.GenerateContentConfig(temperature=0.4),
            )
            motion = resp.text.strip()
            print("🤖 Gemini 자동 생성:", motion)
            return motion
        except Exception as e:
            print(f"⚠️ Gemini 호출 실패({e}) → 기본 프롬프트 사용")

    print("📝 기본 모션 프롬프트 사용:", DEFAULT_MOTION)
    return DEFAULT_MOTION


# ================================================================
# 3. Wan 2.2 I2V 영상 생성 (핵심)
# ================================================================
def _load_gguf_transformer(cfg):
    """GGUF 양자화 transformer를 diffusers로 로드 (VRAM 절감).

    검증된 레시피:
      - gguf 패키지 + GGUFQuantizationConfig 필요.
      - from_single_file은 config를 14B로 오인하므로 해당 모델의 transformer config를
        '명시'해야 한다(로컬 캐시에서 가져옴).
      - 원본(ComfyUI) 키 → diffusers 키 변환은 자동 처리됨.
    """
    import os
    import torch
    from diffusers import WanTransformer3DModel, GGUFQuantizationConfig
    from huggingface_hub import hf_hub_download, snapshot_download

    quant = cfg["gguf_quant"]
    fname = f"Wan2.2-TI2V-5B-{quant}.gguf"
    # 로컬에 미리 받아둔 GGUF가 있으면 재다운로드하지 않는다(느린 네트워크 대비)
    local = os.path.join(cfg.get("gguf_dir") or "/home/sharkey/models/gguf", fname)
    if os.path.exists(local):
        gguf_path = local
        print(f"  로컬 GGUF 사용: {local}")
    else:
        print(f"  GGUF 다운로드: {cfg['gguf_repo']}/{fname}")
        gguf_path = hf_hub_download(cfg["gguf_repo"], fname)
    # 모델의 transformer config 디렉터리(이미 캐시됨 → 즉시 반환)
    config_dir = os.path.join(snapshot_download(cfg["model_id"]), "transformer")
    print(f"  GGUF transformer 로드: {fname}  (config: {cfg['model_id']}/transformer)")
    return WanTransformer3DModel.from_single_file(
        gguf_path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16,
        config=config_dir,
    )


def load_wan_pipeline(cfg):
    print("\n[3] Wan 2.2 I2V 파이프라인 로드 중... (첫 실행 시 모델 다운로드)")
    import torch
    try:
        from diffusers import WanImageToVideoPipeline
    except ImportError as e:
        raise RuntimeError(
            "diffusers에서 WanImageToVideoPipeline을 찾을 수 없습니다. "
            "'pip install -U diffusers>=0.31' 후 다시 시도하세요."
        ) from e

    if cfg.get("gguf_quant"):
        transformer = _load_gguf_transformer(cfg)
        pipe = WanImageToVideoPipeline.from_pretrained(
            cfg["model_id"], transformer=transformer, torch_dtype=torch.bfloat16
        )
    else:
        pipe = WanImageToVideoPipeline.from_pretrained(
            cfg["model_id"], torch_dtype=torch.bfloat16
        )

    pipe.enable_model_cpu_offload()   # VRAM 절약 (12GB에서 권장)
    # VAE 타일링/슬라이싱 — 디코드 단계 OOM 방지 (메모리 안전)
    try:
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
    except Exception:
        pass
    mode = f"GGUF {cfg['gguf_quant']}" if cfg.get("gguf_quant") else "bf16"
    print(f"✅ Wan 2.2 I2V 로드 완료: {cfg['model_id']} ({mode})")
    return pipe


def step_video(cfg, pipe, img_path):
    print("\n[3] 영상 생성 중... (GPU에 따라 수 분 소요)")
    from diffusers.utils import load_image, export_to_video
    image = load_image(img_path)

    result = pipe(
        image=image,
        prompt=cfg["motion_prompt"],
        negative_prompt="blurry, distorted, low quality, artifacts, warping",
        height=cfg["height"],
        width=cfg["width"],
        num_frames=cfg["video_length_frames"],
        num_inference_steps=cfg["video_steps"],
        guidance_scale=cfg["guidance_scale"],
    )
    frames = result.frames[0]

    raw_video = os.path.join(WORK, "03_videos_raw", "raw.mp4")
    export_to_video(frames, raw_video, fps=cfg["video_fps"])
    print("✅ 원본 영상 저장:", raw_video)
    return raw_video


# ================================================================
# 4. 프레임 보간 (RIFE 또는 FFmpeg minterpolate)
# ================================================================
def step_interpolate(cfg, raw_video):
    print("\n[4] 프레임 보간")
    interp_out = os.path.join(WORK, "04_interpolated", "interp.mp4")
    multiplier = max(1, round(cfg["interpolate_to_fps"] / cfg["video_fps"]))

    rife_dir = os.path.join(TOOLS, "Practical-RIFE")
    if cfg["use_rife"]:
        if not os.path.exists(rife_dir):
            print("  Practical-RIFE 클론 중...")
            run(["git", "clone", "-q",
                 "https://github.com/hzwer/Practical-RIFE", rife_dir])
            run([sys.executable, "-m", "pip", "install", "-q", "-r",
                 os.path.join(rife_dir, "requirements.txt")])
            print("  ⚠️ RIFE 모델 가중치(train_log)를 README 안내대로 받아두세요:")
            print("     https://github.com/hzwer/Practical-RIFE")
        train_log = os.path.join(rife_dir, "train_log")
        if os.path.exists(train_log):
            run([sys.executable, "inference_video.py",
                 "--multi", str(multiplier),
                 "--video", raw_video, "--output", interp_out],
                cwd=rife_dir)
            print("✅ RIFE 보간 완료:", interp_out)
            return interp_out
        print("  ⚠️ RIFE 가중치가 없어 FFmpeg minterpolate로 폴백합니다.")

    # FFmpeg minterpolate 폴백
    vf = (f"minterpolate=fps={cfg['interpolate_to_fps']}:"
          f"mi_mode=mci:mc_mode=aobmc:vsbmc=1")
    run(["ffmpeg", "-y", "-i", raw_video, "-vf", vf, interp_out], quiet=True)
    print(f"✅ FFmpeg minterpolate 완료 ({multiplier}x):", interp_out)
    return interp_out


# ================================================================
# 5. 영상 업스케일 (Real-ESRGAN ncnn-vulkan, 선택)
# ================================================================
def step_upscale(cfg, current_video):
    if not cfg["upscale_video"]:
        print("\n[5] 영상 업스케일 건너뜀")
        return current_video

    print("\n[5] 영상 업스케일 (Real-ESRGAN)")
    esrgan_dir = os.path.join(TOOLS, "realesrgan")
    esrgan_bin = os.path.join(esrgan_dir, "realesrgan-ncnn-vulkan")
    if not os.path.exists(esrgan_bin):
        zip_path = os.path.join(TOOLS, "realesrgan.zip")
        url = ("https://github.com/xinntao/Real-ESRGAN/releases/download/"
               "v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip")
        run(["wget", "-q", url, "-O", zip_path])
        run(["unzip", "-q", "-o", zip_path, "-d", esrgan_dir])
        run(["chmod", "+x", esrgan_bin])

    frames_in = os.path.join(WORK, "_frames_in")
    frames_out = os.path.join(WORK, "_frames_out")
    shutil.rmtree(frames_in, ignore_errors=True)
    shutil.rmtree(frames_out, ignore_errors=True)
    os.makedirs(frames_in)
    os.makedirs(frames_out)

    # 프레임 추출 -> 업스케일 -> 재합성
    run(["ffmpeg", "-y", "-i", current_video,
         os.path.join(frames_in, "f_%05d.png")], quiet=True)
    run([esrgan_bin, "-i", frames_in, "-o", frames_out,
         "-n", "realesrgan-x4plus", "-s", str(cfg["upscale_factor"])], quiet=True)

    upscaled = os.path.join(WORK, "05_upscaled_video", "upscaled.mp4")
    run(["ffmpeg", "-y", "-framerate", str(cfg["interpolate_to_fps"]),
         "-i", os.path.join(frames_out, "f_%05d.png"),
         "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", upscaled],
        quiet=True)
    print("✅ 업스케일 완료:", upscaled)
    return upscaled


# ================================================================
# 6. 색보정 + 오디오 (FFmpeg, 최종 출력)
# ================================================================
def step_finalize(cfg, current_video):
    print("\n[6] 색보정 + 오디오 (최종 출력)")
    final_video = os.path.join(WORK, "06_final", "final.mp4")
    eq = (f"eq=contrast={cfg['contrast']}:"
          f"saturation={cfg['saturation']}:"
          f"brightness={cfg['brightness']}")
    vf = f"{eq},fade=in:0:8"

    bgm = cfg.get("bgm_path")
    if bgm and os.path.exists(bgm):
        run(["ffmpeg", "-y", "-i", current_video, "-i", bgm,
             "-vf", vf, "-c:v", "libx264", "-crf", "17", "-preset", "slow",
             "-c:a", "aac", "-b:a", "192k", "-shortest", final_video], quiet=True)
    else:
        run(["ffmpeg", "-y", "-i", current_video,
             "-vf", vf, "-c:v", "libx264", "-crf", "17", "-preset", "slow",
             "-pix_fmt", "yuv420p", final_video], quiet=True)

    print("🎉 최종 영상 저장:", final_video)
    return final_video


# ================================================================
# 메인 오케스트레이션
# ================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Image -> Video Pipeline (local)")
    p.add_argument("--prompt", help="이미지 프롬프트 (CONFIG override)")
    p.add_argument("--motion", help="모션 프롬프트 (영어, 지정 시 Gemini 건너뜀)")
    p.add_argument("--aspect", choices=["16:9", "9:16", "1:1"], help="종횡비")
    p.add_argument("--width", type=int, help="base_width")
    p.add_argument("--steps", type=int, help="추론 스텝")
    p.add_argument("--upscale", action="store_true", help="영상 업스케일 켜기")
    p.add_argument("--gguf", help="GGUF 양자화 사용 (예: Q5_K_M, Q6_K, Q8_0). VRAM 절감")
    return p.parse_args()


def main():
    args = parse_args()
    if args.prompt:
        CONFIG["image_prompt"] = args.prompt
    if args.motion:
        CONFIG["motion_prompt"] = args.motion
    if args.aspect:
        CONFIG["aspect"] = args.aspect
    if args.width:
        CONFIG["base_width"] = args.width
    if args.steps:
        CONFIG["video_steps"] = args.steps
    if args.upscale:
        CONFIG["upscale_video"] = True
    if args.gguf:
        CONFIG["gguf_quant"] = args.gguf

    w, h = resolve_resolution(CONFIG)
    print("=" * 60)
    print(f"해상도: {w} x {h}  |  "
          f"길이: {round(CONFIG['video_length_frames'] / CONFIG['video_fps'], 1)}초  |  "
          f"모델: {CONFIG['model_id']}")
    print("=" * 60)

    ensure_dirs()
    if not check_gpu():
        print("GPU 없이 진행하면 매우 느리거나 실패합니다. 중단합니다.")
        sys.exit(1)

    img_path = step_image(CONFIG)
    CONFIG["motion_prompt"] = step_motion_prompt(CONFIG, img_path)

    pipe = load_wan_pipeline(CONFIG)
    raw_video = step_video(CONFIG, pipe, img_path)

    # VRAM 정리 (다음 단계의 ffmpeg/업스케일 전에)
    try:
        import torch
        del pipe
        torch.cuda.empty_cache()
    except Exception:
        pass

    interp_video = step_interpolate(CONFIG, raw_video)
    upscaled_video = step_upscale(CONFIG, interp_video)
    final_video = step_finalize(CONFIG, upscaled_video)

    print("\n" + "=" * 60)
    print("✅ 파이프라인 완료")
    print("최종 결과:", final_video)
    print("=" * 60)


if __name__ == "__main__":
    main()

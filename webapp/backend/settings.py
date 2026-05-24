"""웹앱 전역 설정 · 작업 경로 · CONFIG 기본값/검증.

원본 CLI(image_to_video_pipeline.py)의 CONFIG 의미를 그대로 따르되,
웹 요청으로 들어온 값을 안전하게 정규화(클램프/선택지 검증)한다.
"""
import os
from pathlib import Path

# 모델 캐시 (원본 스크립트와 동일 — WSL 네이티브 경로 권장)
os.environ.setdefault("HF_HOME", "/home/sharkey/hf_cache")


def _stabilize_downloads():
    """모델 다운로드 안정화 (WSL 환경 대응) — 원본 CLI와 동일한 처리.

    WSL2에서 IPv6 경로가 죽어 있으면 huggingface_hub이 큰 파일에서 멈추므로
    파이썬이 IPv4만 쓰도록 강제하고, 느린 Xet 대신 클래식 CDN을 쓰며,
    HF_HOME 변경으로 가려진 토큰을 직접 주입해 인증(고속) 다운로드를 보장한다.
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

# 작업 디렉터리 (모든 산출물이 여기에 저장됨, 저장소 바깥)
WORK = Path(os.environ.get("IMG2VID_WORK", "/home/sharkey/img2video_work"))
WEBJOBS = WORK / "webjobs"          # 잡별 산출물 루트 (/files 로 정적 서빙)
TOOLS = WORK / "tools"              # RIFE / Real-ESRGAN 등 외부 도구
WEBJOBS.mkdir(parents=True, exist_ok=True)
TOOLS.mkdir(parents=True, exist_ok=True)

# 선택지 (프론트 폼과 동기화)
IMAGE_MODELS = ["flux", "turbo", "flux-realism"]
ASPECTS = ["16:9", "9:16", "1:1"]
WAN_MODELS = [
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers",     # 경량(12GB VRAM)
    "Wan-AI/Wan2.2-I2V-A14B-Diffusers",    # 고품질(VRAM 많이 필요)
]
# GGUF 양자화 선택지 (None=일반 bf16). VRAM 절감용. "" → None 처리
GGUF_QUANTS = ["", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]

DEFAULT_CONFIG = {
    "image_prompt": ("a serene mountain lake at golden hour, cinematic, "
                     "photorealistic, dramatic clouds reflecting on water"),
    "image_model": "flux",
    "image_seed": 42,
    "aspect": "16:9",
    "base_width": 960,            # RTX 3060(12GB) 안전값 (1280은 OOM 위험)
    "model_id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    "gguf_quant": None,           # None=bf16, "Q5_K_M" 등이면 GGUF 양자화(VRAM 절감)
    "gguf_repo": "QuantStack/Wan2.2-TI2V-5B-GGUF",
    "gguf_dir": "/home/sharkey/models/gguf",   # 로컬 GGUF 우선 위치(있으면 재다운로드 안 함)
    "motion_prompt": None,
    "video_length_frames": 49,
    "video_fps": 16,
    "video_steps": 30,
    "guidance_scale": 5.0,
    "interpolate_to_fps": 48,
    "use_rife": False,
    "upscale_video": False,
    "upscale_factor": 2,
    "contrast": 1.08,
    "saturation": 1.12,
    "brightness": 0.02,
    "bgm_path": None,
}


def _num(value, default, lo, hi, cast=float):
    try:
        x = cast(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, x))


def normalize(raw):
    """프론트에서 받은 dict를 안전한 CONFIG로 정규화."""
    raw = raw or {}
    c = dict(DEFAULT_CONFIG)

    # 문자열 / 선택지
    if isinstance(raw.get("image_prompt"), str) and raw["image_prompt"].strip():
        c["image_prompt"] = raw["image_prompt"].strip()
    if raw.get("image_model") in IMAGE_MODELS:
        c["image_model"] = raw["image_model"]
    if raw.get("aspect") in ASPECTS:
        c["aspect"] = raw["aspect"]
    if raw.get("model_id") in WAN_MODELS:
        c["model_id"] = raw["model_id"]

    # GGUF 양자화: 화이트리스트만 허용, ""/없음이면 None(=bf16)
    gq = raw.get("gguf_quant")
    c["gguf_quant"] = gq if (gq in GGUF_QUANTS and gq) else None

    # 시드: None(랜덤) 또는 정수
    seed = raw.get("image_seed", 42)
    if seed in (None, "", "null", "random"):
        c["image_seed"] = None
    else:
        c["image_seed"] = int(_num(seed, 42, 0, 2**31 - 1, int))

    # 모션 프롬프트: 비우면 None(→ Gemini/기본값)
    mp = raw.get("motion_prompt")
    c["motion_prompt"] = mp.strip() if isinstance(mp, str) and mp.strip() else None

    # 수치 (클램프)
    c["base_width"] = (int(_num(raw.get("base_width"), 960, 256, 1536, int)) // 16) * 16
    c["video_length_frames"] = int(_num(raw.get("video_length_frames"), 49, 9, 121, int))
    c["video_fps"] = int(_num(raw.get("video_fps"), 16, 8, 32, int))
    c["video_steps"] = int(_num(raw.get("video_steps"), 30, 1, 60, int))
    c["guidance_scale"] = round(_num(raw.get("guidance_scale"), 5.0, 0.0, 20.0), 2)
    c["interpolate_to_fps"] = int(_num(raw.get("interpolate_to_fps"), 48, 8, 120, int))
    c["use_rife"] = bool(raw.get("use_rife", False))
    c["upscale_video"] = bool(raw.get("upscale_video", False))
    uf = int(_num(raw.get("upscale_factor"), 2, 2, 4, int))
    c["upscale_factor"] = uf if uf in (2, 3, 4) else 2
    c["contrast"] = round(_num(raw.get("contrast"), 1.08, 0.0, 3.0), 3)
    c["saturation"] = round(_num(raw.get("saturation"), 1.12, 0.0, 3.0), 3)
    c["brightness"] = round(_num(raw.get("brightness"), 0.02, -1.0, 1.0), 3)
    c["bgm_path"] = None  # 업로드 시 JobManager가 채움
    return c

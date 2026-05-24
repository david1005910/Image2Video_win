"""원본 CLI 파이프라인(image_to_video_pipeline.py)을 잡(job) 단위로 실행한다.

핵심 아이디어:
  - 원본 스크립트는 모듈 전역 WORK/TOOLS 를 기준으로 산출물을 쓴다.
  - 잡마다 다른 작업 폴더를 쓰려면 import 후 그 전역을 잡 폴더로 갈아끼우면 된다.
  - GPU가 1개뿐이라 잡은 절대 동시 실행하지 않는다(JobManager가 단일 워커로 직렬화).

진행 상황은 progress(stage_key, status) 콜백으로 단계별 보고한다.
"""
import sys
from pathlib import Path

# 저장소 루트(원본 스크립트가 있는 곳)를 import 경로에 추가
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def run_job(job_dir, tools_dir, cfg, progress):
    """한 건의 이미지→영상 파이프라인을 끝까지 실행하고 결과 경로를 돌려준다.

    Args:
        job_dir:   이 잡의 작업 폴더(Path 또는 str). 01_images…06_final 이 여기 생긴다.
        tools_dir: RIFE/Real-ESRGAN 공용 설치 폴더(잡 간 재사용).
        cfg:       settings.normalize() 가 만든 안전한 CONFIG dict.
        progress:  progress(stage_key, status) 콜백. status ∈ {"running","done"}.

    Returns:
        {"final": <final.mp4 경로>, "source": <source.png 경로>}

    Raises:
        RuntimeError: GPU가 없거나 단계 중 하나가 실패하면 그대로 전파한다.
    """
    import image_to_video_pipeline as P

    # 전역 작업 경로를 이 잡 전용으로 교체 (단일 워커라 안전)
    P.WORK = str(job_dir)
    P.TOOLS = str(tools_dir)

    w, h = P.resolve_resolution(cfg)
    print("=" * 60)
    print(f"해상도: {w} x {h}  |  "
          f"길이: {round(cfg['video_length_frames'] / cfg['video_fps'], 1)}초  |  "
          f"모델: {cfg['model_id']}")
    print("=" * 60)
    P.ensure_dirs()

    # GPU 확인 (웹에서는 sys.exit 대신 예외로 알린다)
    import torch
    P.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
           "--format=csv"], check=False)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 찾을 수 없습니다. img2vid 환경/드라이버를 확인하세요.")

    # 1) 이미지
    progress("image", "running")
    img_path = P.step_image(cfg)
    progress("image", "done")

    # 2) 모션 프롬프트
    progress("motion", "running")
    cfg["motion_prompt"] = P.step_motion_prompt(cfg, img_path)
    progress("motion", "done")

    # 3) Wan 2.2 I2V 영상 생성 (핵심)
    progress("video", "running")
    pipe = P.load_wan_pipeline(cfg)
    raw_video = P.step_video(cfg, pipe, img_path)
    # 다음 단계(ffmpeg/업스케일) 전에 VRAM 정리 — 원본 main()의 teardown 보존
    try:
        del pipe
        torch.cuda.empty_cache()
    except Exception:
        pass
    progress("video", "done")

    # 4) 프레임 보간
    progress("interpolate", "running")
    interp_video = P.step_interpolate(cfg, raw_video)
    progress("interpolate", "done")

    # 5) 업스케일 (선택)
    progress("upscale", "running")
    upscaled_video = P.step_upscale(cfg, interp_video)
    progress("upscale", "done")

    # 6) 색보정 + 오디오 (최종)
    progress("finalize", "running")
    final_video = P.step_finalize(cfg, upscaled_video)
    progress("finalize", "done")

    return {"final": final_video, "source": img_path}

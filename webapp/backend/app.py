"""FastAPI 앱 — 이미지→영상 파이프라인 웹 프런트의 백엔드.

엔드포인트:
  GET  /api/options        폼 선택지 + 기본 CONFIG
  POST /api/jobs           잡 생성(멀티파트: config=JSON 문자열, bgm=선택 파일)
  GET  /api/jobs           잡 목록
  GET  /api/jobs/{id}      잡 상세(단계/로그/결과 URL)
  /files/...               잡 산출물 정적 서빙 (WEBJOBS)
  /                         프런트엔드(정적 SPA)

실행(저장소 루트에서, conda activate img2vid 후):
  uvicorn webapp.backend.app:app --host 0.0.0.0 --port 8000
"""
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles

from . import settings
from .jobs import manager

app = FastAPI(title="Image → Video 파이프라인")

_FRONTEND = Path(__file__).resolve().parents[1] / "frontend"


@app.get("/api/options")
def options():
    """프런트 폼이 셀렉트박스/기본값을 채우는 데 쓰는 메타데이터."""
    return {
        "image_models": settings.IMAGE_MODELS,
        "aspects": settings.ASPECTS,
        "wan_models": settings.WAN_MODELS,
        "gguf_quants": settings.GGUF_QUANTS,
        "defaults": settings.DEFAULT_CONFIG,
    }


@app.post("/api/jobs")
async def create_job(
    config: str = Form(...),
    bgm: Optional[UploadFile] = File(None),
):
    """새 잡을 큐에 넣는다. config는 JSON 문자열, bgm은 선택 업로드."""
    try:
        raw = json.loads(config)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="config는 JSON 문자열이어야 합니다.")

    cfg = settings.normalize(raw)
    job = manager.create(cfg)

    # BGM 업로드가 있으면 잡 폴더에 저장하고 CONFIG에 경로 주입
    if bgm is not None and bgm.filename:
        audio_dir = job.dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        dest = audio_dir / Path(bgm.filename).name
        dest.write_bytes(await bgm.read())
        cfg["bgm_path"] = str(dest)

    manager.enqueue(job)
    return {"job_id": job.id}


@app.get("/api/jobs")
def list_jobs():
    return [j.summary() for j in manager.all()]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="해당 잡을 찾을 수 없습니다.")
    return job.detail()


# 잡 산출물 정적 서빙 (final.mp4, source.png 등)
app.mount("/files", StaticFiles(directory=str(settings.WEBJOBS)), name="files")
# 프런트엔드 SPA (반드시 마지막에 마운트 — 루트 경로를 잡아먹으므로)
app.mount("/", StaticFiles(directory=str(_FRONTEND), html=True), name="frontend")

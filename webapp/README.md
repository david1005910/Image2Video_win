# Image → Video 웹앱

CLI 파이프라인(`image_to_video_pipeline.py`)을 그대로 재사용하는 풀스택 웹 프런트.
빌드 도구 없이 **FastAPI 백엔드 + 정적 바닐라 JS 프런트**로 구성된다.

## 구조

```
webapp/
  backend/
    settings.py          # CONFIG 기본값 + 웹 입력 정규화/클램프 (원본 CONFIG와 동기화 필요)
    pipeline_runner.py   # 원본 step_* 함수를 잡 폴더 기준으로 실행
    jobs.py              # 단일 GPU 워커 + 큐 + 상태/로그 (JobManager)
    app.py               # FastAPI 엔드포인트 + 정적 서빙
  frontend/
    index.html / style.css / app.js   # 설정 폼 + 진행바 + 결과 영상
  requirements.txt
  run.sh
```

## 실행

```bash
conda activate img2vid                       # 파이프라인 의존성이 있는 환경
pip install -r webapp/requirements.txt       # 최초 1회 (fastapi/uvicorn/python-multipart)
./webapp/run.sh                              # → http://localhost:8000
# 또는: uvicorn webapp.backend.app:app --host 0.0.0.0 --port 8000  (저장소 루트에서)
```

브라우저에서 `http://localhost:8000` 접속 → 설정 입력 → "생성 시작".
잡은 GPU가 1개라 **순차 처리**되며, 진행 단계가 실시간 폴링으로 표시되고
완료되면 결과 영상이 페이지에 임베드된다.

## 동작 메모

- 모든 산출물은 `$IMG2VID_WORK/webjobs/<job_id>/` 아래(`01_images`…`06_final`)에 쌓이고
  `/files/...` 로 정적 서빙된다. RIFE/Real-ESRGAN 도구는 `$IMG2VID_WORK/tools` 에서 잡 간 공유.
- `GEMINI_API_KEY` 가 설정돼 있으면 모션 프롬프트를 비웠을 때 자동 생성된다.
- 잡 상태/로그는 메모리에만 있으므로 서버를 재시작하면 목록이 사라진다(산출물 파일은 남음).
- 단일 사용자 로컬 도구 전제다(인증/동시성 보호 없음).

## API

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET  | `/api/options`      | 폼 선택지 + 기본 CONFIG |
| POST | `/api/jobs`         | 잡 생성 (멀티파트: `config`=JSON, `bgm`=선택 파일) |
| GET  | `/api/jobs`         | 잡 목록 |
| GET  | `/api/jobs/{id}`    | 잡 상세(단계/로그/결과 URL) |

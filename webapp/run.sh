#!/usr/bin/env bash
# 웹앱 실행 스크립트.
#   1) conda activate img2vid   (먼저!)
#   2) 최초 1회: pip install -r webapp/requirements.txt
#   3) ./webapp/run.sh
# 저장소 루트에서 uvicorn 을 띄워야 image_to_video_pipeline.py 를 import 할 수 있다.
set -e
cd "$(dirname "$0")/.."   # 저장소 루트로 이동
exec uvicorn webapp.backend.app:app --host 0.0.0.0 --port 8000 "$@"

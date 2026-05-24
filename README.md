# Image2Video_win

텍스트 프롬프트 한 줄을 짧은 영상 클립으로 만드는 **이미지→영상 생성 파이프라인**.
무료/저비용 서비스와 로컬 모델을 엮어 GPU에서 돌아간다. **WSL2 + conda + RTX 3060** 환경에 맞춰
포팅된 로컬 버전이며, 같은 파이프라인을 감싸는 웹 프런트엔드(`webapp/`)도 포함한다.

```
Pollinations(이미지) → Gemini(모션 프롬프트, 선택) → Wan 2.2 I2V(영상)
  → RIFE / FFmpeg(프레임 보간) → Real-ESRGAN(업스케일, 선택) → FFmpeg(색보정 / 오디오)
```

## 구성

| 파일 | 설명 |
|------|------|
| `image_to_video_pipeline.py` | **유지보수 대상.** 로컬(WSL) CLI 파이프라인. |
| `image_to_video_pipeline.ipynb` | 원본 Google Colab 노트북(T4 GPU). 참고용 — ComfyUI 경로와 Veo 3.1 하이브리드 경로를 추가로 문서화. |
| `webapp/` | 같은 파이프라인을 쓰는 풀스택 웹 프런트(FastAPI + 바닐라 JS). [`webapp/README.md`](webapp/README.md) 참고. |

## 파이프라인 단계

`main()`이 6개 단계를 순차 실행하며, 각 단계는 작업 폴더의 번호 붙은 하위 디렉터리(`01_images` … `06_final`)에 산출물을 쓴다.

1. **이미지** — Pollinations 무료 API에서 PNG 생성 (API 키 불필요).
2. **모션 프롬프트** — `GEMINI_API_KEY`가 있으면 Gemini Vision으로 자동 생성, 없으면 기본값. (직접 지정도 가능)
3. **영상** — Wan 2.2 I2V (diffusers `WanImageToVideoPipeline`, bf16, CPU offload). 파이프라인의 핵심.
4. **프레임 보간** — Practical-RIFE(가중치 있을 때) 또는 FFmpeg `minterpolate` 폴백.
5. **업스케일** — Real-ESRGAN(선택). WSL의 Vulkan은 불안정하므로 기본 꺼짐.
6. **마무리** — FFmpeg 색보정 + 페이드인, 선택적 BGM 합성 → `06_final/final.mp4`.

## 요구 사항

- **WSL2** + conda 환경 `img2vid` (CUDA 12.8 빌드 PyTorch)
- **CUDA GPU** 필수 — GPU가 없으면 실행이 중단된다.
- PATH에 `ffmpeg`(필수), `git`/`wget`/`unzip`/`chmod`(RIFE·Real-ESRGAN 사용 시)

## 설치 (최초 1회)

```bash
conda activate img2vid
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install "diffusers>=0.31" transformers accelerate safetensors ftfy \
            imageio imageio-ffmpeg requests pillow
pip install google-genai          # (선택) Gemini 모션 프롬프트 자동 생성용
pip install gguf                   # (선택) GGUF 양자화 사용 시
sudo apt update && sudo apt install -y ffmpeg
hf auth login                      # HuggingFace 모델 다운로드 인증
```

## 실행

```bash
conda activate img2vid
python image_to_video_pipeline.py

# CONFIG 값을 덮어쓰는 플래그:
python image_to_video_pipeline.py --prompt "a neon city at night" --aspect 9:16 --steps 30 --upscale
python image_to_video_pipeline.py --gguf Q5_K_M    # GGUF 양자화 transformer (VRAM 대폭 절감)
```

사용 가능한 플래그: `--prompt` `--motion` `--aspect {16:9,9:16,1:1}` `--width` `--steps` `--upscale` `--gguf`

GPU 없이 빠르게 확인(인자 파싱/임포트만):

```bash
python image_to_video_pipeline.py --help
```

## 설정

모든 튜닝 값은 `.py` 상단의 `CONFIG` 딕셔너리에 있다. CLI 플래그는 일부 키만 덮어쓰고, 나머지는 `CONFIG`에서 직접 수정한다.

- **`base_width`** — VRAM 핵심 레버. 12GB 3060은 832~1024 권장 (1280은 OOM 위험).
- **`gguf_quant`** — GGUF 양자화 transformer (`Q5_K_M` 등). bf16 ~10GB → Q5 ~3.8GB로 VRAM 대폭 절감.
- **`model_id`** — 경량 `Wan2.2-TI2V-5B` ↔ 고품질 `Wan2.2-I2V-A14B` 교체.

### 환경 변수

| 변수 | 설명 |
|------|------|
| `IMG2VID_WORK` | 작업 폴더 (기본 `/home/sharkey/img2video_work`). **모든 산출물이 저장소 바깥의 이 경로에 쌓인다.** |
| `HF_HOME` | HuggingFace 모델 캐시 (기본 `/home/sharkey/hf_cache`). |
| `GEMINI_API_KEY` | (선택) 모션 프롬프트 자동 생성 활성화. 코드에 하드코딩하지 말 것. |
| `HF_TOKEN` | 미설정 시 캐시 토큰에서 자동 주입됨. |

> **WSL 다운로드 안정화:** 임포트 시 IPv4 강제, Xet 백엔드 비활성화, `HF_TOKEN` 주입을 자동 수행한다(WSL2의 IPv6가 큰 HF 다운로드를 멈추는 문제 회피). 이는 의도된 우회 처리다.

## 웹 앱

CLI 파이프라인을 그대로 재사용하는 빌드리스 웹 프런트엔드.

```bash
conda activate img2vid
pip install -r webapp/requirements.txt   # 최초 1회 (fastapi / uvicorn / python-multipart)
./webapp/run.sh                          # → http://localhost:8000
```

GPU가 1개라 잡은 순차 처리되며, 진행 단계가 실시간으로 표시되고 완료 시 결과 영상이 임베드된다.
자세한 내용은 [`webapp/README.md`](webapp/README.md) 참고.

## 참고

- 코드의 주석·docstring·콘솔 출력은 **한국어**로 작성되어 있다.
- 테스트 스위트나 빌드 시스템은 없다. 변경 검증은 실제 실행 또는 위의 GPU 없는 드라이체크로 한다.
- 노트북(`.ipynb`)은 참고용이며, `.py`가 생략한 ComfyUI / Veo 3.1 하이브리드 경로를 추가로 다룬다.

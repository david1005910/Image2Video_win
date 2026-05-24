"""잡(job) 관리: 단일 GPU 워커 + 큐 + 상태/로그 추적.

GPU가 1개라 Wan 파이프라인을 동시에 두 개 올리면 OOM 난다. 그래서 잡은
하나의 백그라운드 워커 스레드가 큐에서 꺼내 '직렬'로 처리한다.
상태/로그는 메모리에 들고 있고(/api/jobs/{id} 로 폴링), 산출물은
WEBJOBS/<job_id>/ 아래에 쌓여 /files 로 정적 서빙된다.
"""
import sys
import time
import uuid
import queue
import threading
import traceback

from . import settings
from .pipeline_runner import run_job

# 파이프라인 단계 정의 (키, 표시 이름) — 프론트 진행바와 1:1 대응
STAGES = [
    ("image", "이미지 생성"),
    ("motion", "모션 프롬프트"),
    ("video", "영상 생성 (Wan 2.2)"),
    ("interpolate", "프레임 보간"),
    ("upscale", "업스케일"),
    ("finalize", "색보정 / 오디오"),
]


class Job:
    """잡 1건의 상태. 메모리에만 존재(서버 재시작 시 사라짐)."""

    def __init__(self, job_id, job_dir, cfg):
        self.id = job_id
        self.dir = job_dir          # Path
        self.cfg = cfg
        self.status = "queued"      # queued | running | done | error
        self.error = None
        self.created = time.strftime("%Y-%m-%d %H:%M:%S")
        self.log = []               # 최근 콘솔 출력 라인들
        self.stages = [
            {"key": k, "label": label, "status": "pending"}  # pending|running|done
            for k, label in STAGES
        ]

    def set_stage(self, key, status):
        for s in self.stages:
            if s["key"] == key:
                s["status"] = status
                break

    def _url(self, rel, must_exist=True):
        if must_exist and not (self.dir / rel).exists():
            return None
        return f"/files/{self.id}/{rel}"

    def summary(self):
        prompt = (self.cfg.get("image_prompt") or "")[:80]
        return {
            "id": self.id,
            "status": self.status,
            "created": self.created,
            "prompt": prompt,
        }

    def detail(self):
        d = self.summary()
        d.update({
            "error": self.error,
            "stages": self.stages,
            "log": self.log[-80:],
            "source_url": self._url("01_images/source.png"),
            "final_url": self._url("06_final/final.mp4"),
            "config": {
                "aspect": self.cfg.get("aspect"),
                "base_width": self.cfg.get("base_width"),
                "model_id": self.cfg.get("model_id"),
                "video_steps": self.cfg.get("video_steps"),
                "motion_prompt": self.cfg.get("motion_prompt"),
            },
        })
        return d


class _LogTee:
    """파이프라인의 print 출력을 잡 로그에도, 원래 콘솔에도 함께 흘린다."""

    def __init__(self, job, original):
        self.job = job
        self.original = original

    def write(self, s):
        self.original.write(s)
        for line in s.splitlines():
            if line.strip():
                self.job.log.append(line)
        if len(self.job.log) > 500:
            del self.job.log[:-500]

    def flush(self):
        self.original.flush()


class JobManager:
    def __init__(self):
        self._jobs = {}
        self._order = []
        self._q = queue.Queue()
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def create(self, cfg):
        """잡 폴더를 만들고 등록한다(아직 큐에는 안 넣음 — BGM 저장 후 enqueue)."""
        jid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        job_dir = settings.WEBJOBS / jid
        job_dir.mkdir(parents=True, exist_ok=True)
        job = Job(jid, job_dir, cfg)
        with self._lock:
            self._jobs[jid] = job
            self._order.append(jid)
        return job

    def enqueue(self, job):
        self._q.put(job.id)

    def get(self, jid):
        return self._jobs.get(jid)

    def all(self):
        with self._lock:
            return [self._jobs[i] for i in reversed(self._order)]

    def _loop(self):
        while True:
            jid = self._q.get()
            job = self._jobs.get(jid)
            if job:
                self._process(job)
            self._q.task_done()

    def _process(self, job):
        job.status = "running"
        old_out, old_err = sys.stdout, sys.stderr
        tee = _LogTee(job, old_out)
        sys.stdout = sys.stderr = tee  # 단일 워커라 전역 교체로 충분

        def progress(key, status):
            job.set_stage(key, status)

        try:
            run_job(job.dir, settings.TOOLS, job.cfg, progress)
            job.status = "done"
        except Exception:
            job.status = "error"
            job.error = traceback.format_exc()
            # 진행 중이던 단계를 표시상 멈춤 처리
            for s in job.stages:
                if s["status"] == "running":
                    s["status"] = "pending"
            print("❌ 잡 실패:\n" + job.error)
        finally:
            sys.stdout, sys.stderr = old_out, old_err


# 전역 매니저 (앱 전체에서 공유)
manager = JobManager()

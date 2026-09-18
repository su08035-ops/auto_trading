"""Local web server for running experiments from the frontend."""

from __future__ import annotations

import json
import mimetypes
import threading
import time
import traceback
import uuid
from copy import deepcopy
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import sys


BACKEND_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = BACKEND_ROOT.parent
FRONTEND_ROOT = APP_ROOT / "frontend"
RESULTS_ROOT = BACKEND_ROOT / "results"

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from data.loader import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from main import run_experiment  # noqa: E402
from tools.export_frontend_runs import export_frontend_runs  # noqa: E402


JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def _json_bytes(payload) -> bytes:
    return json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")


def _read_body(handler: SimpleHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0"))
    if length == 0:
        return {}
    return json.loads(handler.rfile.read(length).decode("utf-8"))


def _as_int(payload: dict, key: str, default: int, *, minimum: int, maximum: int) -> int:
    value = int(payload.get(key, default))
    if not minimum <= value <= maximum:
        raise ValueError(f"{key} must be between {minimum} and {maximum}.")
    return value


def _as_choice(payload: dict, key: str, default: str, choices: set[str]) -> str:
    value = str(payload.get(key, default))
    if value not in choices:
        raise ValueError(f"{key} must be one of {sorted(choices)}.")
    return value


def _as_date(payload: dict, key: str, default: str) -> str:
    value = str(payload.get(key, default)).strip()
    if len(value) != 10 or value[4] != "-" or value[7] != "-":
        raise ValueError(f"{key} must be YYYY-MM-DD.")
    return value


def build_config(payload: dict) -> dict:
    config = deepcopy(load_config(DEFAULT_CONFIG_PATH))
    config["data"]["source"] = _as_choice(
        payload, "source", config["data"]["source"], {"kis", "synthetic"}
    )
    config["agent"]["algorithm"] = _as_choice(
        payload, "algorithm", config["agent"]["algorithm"], {"dqn", "double_dqn"}
    )
    config["training"]["episodes"] = _as_int(
        payload, "episodes", config["training"]["episodes"], minimum=1, maximum=1000
    )
    config["training"]["seed"] = _as_int(
        payload, "seed", config["training"]["seed"], minimum=0, maximum=999999
    )
    config["environment"]["min_holding_days"] = _as_int(
        payload,
        "min_holding_days",
        config["environment"]["min_holding_days"],
        minimum=0,
        maximum=60,
    )
    for key in ("start_date", "validation_start", "split_date", "end_date"):
        config["data"][key] = _as_date(payload, key, config["data"][key])
    return config


def set_job(job_id: str, **updates) -> None:
    with JOBS_LOCK:
        JOBS[job_id].update(updates)


def append_log(job_id: str, message: str) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        job["logs"].append(message)
        if len(job["logs"]) > 300:
            job["logs"] = job["logs"][-300:]
        if message.startswith("Episode "):
            head = message.split("|", 1)[0].strip()
            current, total = head.replace("Episode ", "").split("/", 1)
            job["progress"] = max(job["progress"], int(current) / max(1, int(total)))


def run_job(job_id: str, config: dict) -> None:
    set_job(job_id, status="running", phase="학습 준비", started_at=time.time())
    try:
        def progress(message: str) -> None:
            append_log(job_id, message)
            if message.startswith("Loading "):
                set_job(job_id, phase="데이터 불러오는 중")
            elif message.startswith("train:"):
                set_job(job_id, phase="기간 분할 완료")
            elif message.startswith("Episode "):
                set_job(job_id, phase="학습 중")
            elif message.startswith("Complete:"):
                set_job(job_id, phase="결과 저장 완료", progress=1.0)

        run_dir = run_experiment(config, progress=progress)
        export_frontend_runs(results_dir=RESULTS_ROOT, frontend_dir=FRONTEND_ROOT)
        set_job(
            job_id,
            status="done",
            phase="완료",
            progress=1.0,
            run_id=run_dir.name,
            finished_at=time.time(),
        )
    except Exception as exc:
        append_log(job_id, traceback.format_exc(limit=8))
        set_job(
            job_id,
            status="error",
            phase="오류",
            error=f"{type(exc).__name__}: {exc}",
            finished_at=time.time(),
        )


class FrontendHandler(SimpleHTTPRequestHandler):
    server_version = "AutoTradingFrontend/0.1"

    def log_message(self, format, *args):
        return

    def send_json(self, payload, status=HTTPStatus.OK):
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, message, status=HTTPStatus.BAD_REQUEST):
        self.send_json({"error": message}, status=status)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/config":
            self.send_json(load_config(DEFAULT_CONFIG_PATH))
            return
        if parsed.path == "/api/runs":
            output = export_frontend_runs(results_dir=RESULTS_ROOT, frontend_dir=FRONTEND_ROOT)
            self.send_json(json.loads(output.read_text(encoding="utf-8")))
            return
        if parsed.path.startswith("/api/jobs/"):
            job_id = parsed.path.rsplit("/", 1)[-1]
            with JOBS_LOCK:
                job = deepcopy(JOBS.get(job_id))
            if job is None:
                self.send_error_json("Job not found.", HTTPStatus.NOT_FOUND)
                return
            self.send_json(job)
            return
        if parsed.path == "/":
            self.serve_file(FRONTEND_ROOT / "index.html")
            return
        self.serve_static(parsed.path)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/experiments":
            self.send_error_json("Unknown endpoint.", HTTPStatus.NOT_FOUND)
            return
        try:
            payload = _read_body(self)
            config = build_config(payload)
        except Exception as exc:
            self.send_error_json(str(exc))
            return
        job_id = uuid.uuid4().hex[:12]
        with JOBS_LOCK:
            JOBS[job_id] = {
                "id": job_id,
                "status": "queued",
                "phase": "대기 중",
                "progress": 0.0,
                "logs": [],
                "run_id": None,
                "error": None,
            }
        thread = threading.Thread(target=run_job, args=(job_id, config), daemon=True)
        thread.start()
        self.send_json({"job_id": job_id}, status=HTTPStatus.ACCEPTED)

    def serve_static(self, path: str):
        clean = unquote(path).lstrip("/")
        if clean.startswith("frontend/"):
            self.serve_file(APP_ROOT / clean)
            return
        if clean.startswith("results/"):
            self.serve_file(BACKEND_ROOT / clean)
            return
        self.send_error_json("File not found.", HTTPStatus.NOT_FOUND)

    def serve_file(self, path: Path):
        resolved = path.resolve()
        allowed_roots = (FRONTEND_ROOT.resolve(), RESULTS_ROOT.resolve())
        if not any(resolved == root or root in resolved.parents for root in allowed_roots):
            self.send_error_json("File not allowed.", HTTPStatus.FORBIDDEN)
            return
        if not resolved.exists() or not resolved.is_file():
            self.send_error_json("File not found.", HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(str(resolved))[0] or "application/octet-stream"
        body = resolved.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    export_frontend_runs(results_dir=RESULTS_ROOT, frontend_dir=FRONTEND_ROOT)
    port = 8000
    server = ThreadingHTTPServer(("127.0.0.1", port), FrontendHandler)
    print(f"API: http://127.0.0.1:{port}")
    print("Frontend: cd frontend && npm run dev")
    print("Stop: Ctrl+C")
    server.serve_forever()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel
from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem


CONFIG: dict[str, Any] = {}
MODEL: Qwen3TTSModel | None = None
VOICE_PROMPT: list[VoiceClonePromptItem] | None = None
VOICE_CACHE: dict[str, list[VoiceClonePromptItem]] = {}
LOCK = threading.Lock()
ACTIVE_LOCK = threading.Lock()
ACTIVE_REQUEST: dict[str, Any] | None = None
QUEUE_LOCK = threading.Lock()
QUEUED_REQUESTS: dict[str, dict[str, Any]] = {}
REJECTED_REQUESTS = 0
LAST_REJECTED_REQUEST: dict[str, Any] | None = None
SYNTHESIS_THREAD: threading.Thread | None = None
LOGGER = logging.getLogger("qwen_local_tts_worker")
app = FastAPI(title="Qwen Local TTS Worker")


class SynthesizeRequest(BaseModel):
    text: str
    language: str = "Auto"
    tone: str = ""
    voice_file: str = ""
    voice_key: str = ""
    reference_audio_file: str = ""
    reference_text: str = ""
    x_vector_only_mode: Optional[bool] = None


class VoiceConfigRequest(BaseModel):
    voice_file: str = ""
    reference_audio_file: str = ""
    reference_text: str = ""
    x_vector_only_mode: bool = False


@dataclass
class SynthesisJob:
    request_id: str
    request: SynthesizeRequest
    created_at: float
    done: threading.Event = field(default_factory=threading.Event)
    result: bytes | None = None
    error: Exception | None = None


JOB_QUEUE: queue.Queue[SynthesisJob] = queue.Queue()


def dtype_from_str(value: str) -> torch.dtype:
    value = (value or "bfloat16").lower()
    if value in ("bf16", "bfloat16"):
        return torch.bfloat16
    if value in ("fp16", "float16"):
        return torch.float16
    if value in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def bool_from_config(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("1", "true", "yes", "y", "on", "enable", "enabled"):
            return True
        if normalized in ("0", "false", "no", "n", "off", "disable", "disabled"):
            return False
    return bool(value)


def int_from_config(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def configured_max_queue_size() -> int:
    return max(1, int_from_config(CONFIG.get("max_queue_size"), 4))


def debug_enabled() -> bool:
    return bool_from_config(CONFIG.get("debug_logging"), False)


def debug_log(message: str, *args: Any) -> None:
    if debug_enabled():
        LOGGER.debug("[QwenWorker] " + message, *args)


def require_worker_token(request: Request) -> None:
    expected = str(CONFIG.get("worker_token") or "").strip()
    if expected and request.headers.get("X-Qwen-Worker-Token") != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


def set_active_request(
    request_id: str,
    text_len: int,
    language: str,
    source: str,
    tone: str = "",
    voice_file_name: str = "",
) -> None:
    global ACTIVE_REQUEST
    with ACTIVE_LOCK:
        ACTIVE_REQUEST = {
            "request_id": request_id,
            "text_len": text_len,
            "language": language,
            "tone": tone,
            "source": source,
            "voice_file_name": voice_file_name,
            "started_at": time.monotonic(),
        }


def clear_active_request(request_id: str) -> None:
    global ACTIVE_REQUEST
    with ACTIVE_LOCK:
        if ACTIVE_REQUEST and ACTIVE_REQUEST.get("request_id") == request_id:
            ACTIVE_REQUEST = None


def active_request_snapshot() -> dict[str, Any] | None:
    with ACTIVE_LOCK:
        if not ACTIVE_REQUEST:
            return None
        snapshot = dict(ACTIVE_REQUEST)
    started_at = float(snapshot.pop("started_at", time.monotonic()))
    snapshot["elapsed"] = round(time.monotonic() - started_at, 3)
    return snapshot


def queue_request_snapshot(job: SynthesisJob) -> dict[str, Any]:
    req = job.request
    voice_file = str(req.voice_file or "").strip()
    return {
        "request_id": job.request_id,
        "text_len": len((req.text or "").strip()),
        "language": (req.language or CONFIG.get("language") or "Auto").strip() or "Auto",
        "tone": (req.tone or "").strip(),
        "voice_key": (req.voice_key or "").strip(),
        "voice_file_name": os.path.basename(voice_file) if voice_file else "",
        "created_at": job.created_at,
    }


def record_queue_rejection(job: SynthesisJob, queued_count: int, max_queue_size: int) -> None:
    global REJECTED_REQUESTS, LAST_REJECTED_REQUEST
    with QUEUE_LOCK:
        REJECTED_REQUESTS += 1
        snapshot = queue_request_snapshot(job)
        snapshot["queued_count"] = queued_count
        snapshot["max_queue_size"] = max_queue_size
        snapshot["rejected_at"] = time.monotonic()
        LAST_REJECTED_REQUEST = snapshot


def try_add_queued_request(job: SynthesisJob) -> tuple[bool, int, int]:
    max_queue_size = configured_max_queue_size()
    with QUEUE_LOCK:
        queued_count = len(QUEUED_REQUESTS)
        if queued_count >= max_queue_size:
            return False, queued_count, max_queue_size
        QUEUED_REQUESTS[job.request_id] = queue_request_snapshot(job)
        return True, queued_count + 1, max_queue_size


def remove_queued_request(request_id: str) -> None:
    with QUEUE_LOCK:
        QUEUED_REQUESTS.pop(request_id, None)


def queued_requests_snapshot() -> list[dict[str, Any]]:
    now = time.monotonic()
    with QUEUE_LOCK:
        snapshots = [dict(item) for item in QUEUED_REQUESTS.values()]
    for item in snapshots:
        created_at = float(item.pop("created_at", now))
        item["queued_for"] = round(now - created_at, 3)
    return snapshots


def queue_rejection_snapshot() -> dict[str, Any]:
    now = time.monotonic()
    with QUEUE_LOCK:
        rejected_count = REJECTED_REQUESTS
        last_rejected = dict(LAST_REJECTED_REQUEST or {})
    if last_rejected:
        rejected_at = float(last_rejected.pop("rejected_at", now))
        last_rejected["rejected_for"] = round(now - rejected_at, 3)
    return {
        "rejected_requests": rejected_count,
        "last_rejected_request": last_rejected or None,
    }


def voice_config_snapshot() -> dict[str, Any]:
    voice_file = str(CONFIG.get("voice_file") or "")
    reference_audio_file = str(CONFIG.get("reference_audio_file") or "")
    return {
        "voice_file": bool(voice_file),
        "voice_file_path": voice_file,
        "voice_file_name": os.path.basename(voice_file) if voice_file else "",
        "reference_audio_file": bool(reference_audio_file),
        "reference_audio_path": reference_audio_file,
        "reference_audio_name": os.path.basename(reference_audio_file) if reference_audio_file else "",
        "voice_prompt_items": len(VOICE_PROMPT or []),
    }


def is_url_path(path: str) -> bool:
    return path.startswith(("http://", "https://"))


def validate_readable_path(path: str, label: str) -> None:
    if path and not is_url_path(path) and not os.path.exists(path):
        raise HTTPException(status_code=400, detail=f"{label} does not exist: {path}")


def tensor_or_none(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value
    return torch.tensor(value)


def load_voice_prompt(path: str) -> list[VoiceClonePromptItem]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "items" not in payload:
        raise ValueError("Invalid Qwen voice file: missing items.")
    items_raw = payload["items"]
    if not isinstance(items_raw, list) or not items_raw:
        raise ValueError("Invalid Qwen voice file: items is empty.")

    items: list[VoiceClonePromptItem] = []
    for raw in items_raw:
        if not isinstance(raw, dict):
            raise ValueError("Invalid Qwen voice item.")
        ref_spk = tensor_or_none(raw.get("ref_spk_embedding"))
        if ref_spk is None:
            raise ValueError("Invalid Qwen voice item: missing ref_spk_embedding.")
        items.append(
            VoiceClonePromptItem(
                ref_code=tensor_or_none(raw.get("ref_code")),
                ref_spk_embedding=ref_spk,
                x_vector_only_mode=bool(raw.get("x_vector_only_mode", False)),
                icl_mode=bool(raw.get("icl_mode", not bool(raw.get("x_vector_only_mode", False)))),
                ref_text=raw.get("ref_text"),
            )
        )
    return items


def wav_bytes(wav: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.asarray(wav, dtype=np.float32), sr, format="WAV")
    return buf.getvalue()


def cached_voice_prompt(path: str) -> list[VoiceClonePromptItem]:
    path = (path or "").strip()
    if not path:
        raise ValueError("voice_file is empty.")
    if path not in VOICE_CACHE:
        VOICE_CACHE[path] = load_voice_prompt(path)
    return VOICE_CACHE[path]


def prepare_voice_for_request(
    req: SynthesizeRequest,
) -> tuple[list[VoiceClonePromptItem] | None, str, str | None, bool, str, str]:
    global VOICE_PROMPT
    request_voice_file = (req.voice_file or "").strip()
    request_reference_audio = (req.reference_audio_file or "").strip()
    request_reference_text = (req.reference_text or "").strip()
    voice_file = request_voice_file or str(CONFIG.get("voice_file") or "").strip()
    reference_audio = request_reference_audio or str(CONFIG.get("reference_audio_file") or "").strip()
    reference_text = request_reference_text or str(CONFIG.get("reference_text") or "").strip()
    if req.x_vector_only_mode is None:
        use_xvec = bool(CONFIG.get("x_vector_only_mode", False))
    else:
        use_xvec = bool(req.x_vector_only_mode)

    validate_readable_path(voice_file, "voice_file")
    validate_readable_path(reference_audio, "reference_audio_file")

    with LOCK:
        if voice_file:
            voice_prompt = cached_voice_prompt(voice_file)
            VOICE_PROMPT = voice_prompt
            CONFIG["voice_file"] = voice_file
            CONFIG["reference_audio_file"] = reference_audio
            CONFIG["reference_text"] = reference_text
            CONFIG["x_vector_only_mode"] = use_xvec
            return voice_prompt, reference_audio, reference_text or None, use_xvec, "voice_prompt", voice_file

        VOICE_PROMPT = None
        CONFIG["voice_file"] = ""
        CONFIG["reference_audio_file"] = reference_audio
        CONFIG["reference_text"] = reference_text
        CONFIG["x_vector_only_mode"] = use_xvec
        return None, reference_audio, reference_text or None, use_xvec, "reference_audio", ""


def synthesize_audio(req: SynthesizeRequest, request_id: str) -> bytes:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty.")

    started_at = time.monotonic()
    language = (req.language or CONFIG.get("language") or "Auto").strip() or "Auto"
    tone = (req.tone or "").strip()
    generation = dict(CONFIG.get("generation") or {})
    try:
        voice_prompt, ref_audio, ref_text, use_xvec, source, voice_file = prepare_voice_for_request(req)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}") from exc

    voice_file_name = os.path.basename(voice_file) if voice_file else ""
    debug_log(
        "request %s synthesize start text_len=%s language=%s tone=%s source=%s voice_key=%s voice_file=%s",
        request_id,
        len(text),
        language,
        tone,
        source,
        req.voice_key or "",
        voice_file_name,
    )

    set_active_request(request_id, len(text), language, source, tone, voice_file_name)
    try:
        if voice_prompt:
            wavs, sr = MODEL.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=voice_prompt,
                **generation,
            )
        else:
            if not ref_audio:
                raise ValueError("voice_file or reference_audio_file must be configured.")
            if not use_xvec and not ref_text:
                raise ValueError("reference_text is required unless x_vector_only_mode is enabled.")
            wavs, sr = MODEL.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=use_xvec,
                **generation,
            )
    except Exception as exc:
        if debug_enabled():
            LOGGER.exception("[QwenWorker] request %s synthesize failed", request_id)
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    finally:
        clear_active_request(request_id)

    audio = wav_bytes(wavs[0], sr)
    debug_log(
        "request %s synthesize done sr=%s wavs=%s bytes=%s elapsed=%.3fs",
        request_id,
        sr,
        len(wavs),
        len(audio),
        time.monotonic() - started_at,
    )
    return audio


def synthesis_worker_loop() -> None:
    while True:
        job = JOB_QUEUE.get()
        remove_queued_request(job.request_id)
        try:
            job.result = synthesize_audio(job.request, job.request_id)
        except Exception as exc:
            job.error = exc
        finally:
            job.done.set()
            JOB_QUEUE.task_done()


def start_synthesis_worker() -> None:
    global SYNTHESIS_THREAD
    if SYNTHESIS_THREAD and SYNTHESIS_THREAD.is_alive():
        return
    SYNTHESIS_THREAD = threading.Thread(
        target=synthesis_worker_loop,
        name="qwen-local-tts-fifo",
        daemon=True,
    )
    SYNTHESIS_THREAD.start()


@app.on_event("startup")
def startup() -> None:
    global MODEL, VOICE_PROMPT
    model_id = CONFIG.get("model") or "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    started_at = time.monotonic()
    debug_log(
        "startup begin model=%s device=%s dtype=%s attn=%s voice_file=%s reference_audio=%s",
        model_id,
        CONFIG.get("device") or "mps",
        CONFIG.get("dtype") or "bfloat16",
        CONFIG.get("attn_implementation") or "sdpa",
        bool(CONFIG.get("voice_file")),
        bool(CONFIG.get("reference_audio_file")),
    )
    MODEL = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map=CONFIG.get("device") or "mps",
        dtype=dtype_from_str(CONFIG.get("dtype") or "bfloat16"),
        attn_implementation=CONFIG.get("attn_implementation") or "sdpa",
    )
    voice_file = CONFIG.get("voice_file") or ""
    if voice_file:
        debug_log("loading voice prompt file")
        VOICE_PROMPT = cached_voice_prompt(voice_file)
        debug_log("voice prompt loaded items=%s", len(VOICE_PROMPT))
    start_synthesis_worker()
    debug_log("startup complete elapsed=%.3fs", time.monotonic() - started_at)


@app.get("/health")
def health(request: Request) -> dict[str, Any]:
    require_worker_token(request)
    active_request = active_request_snapshot()
    queued_requests = queued_requests_snapshot()
    return {
        "ok": MODEL is not None,
        "model": CONFIG.get("model"),
        "debug_logging": debug_enabled(),
        "fingerprint": CONFIG.get("fingerprint", ""),
        "busy": active_request is not None,
        "active_request": active_request,
        "max_queue_size": configured_max_queue_size(),
        "queued_count": len(queued_requests),
        "queued_requests": queued_requests,
        **queue_rejection_snapshot(),
        **voice_config_snapshot(),
    }


@app.post("/voice_config")
def update_voice_config(req: VoiceConfigRequest, request: Request) -> dict[str, Any]:
    """Hot-load the active voice without reloading the Qwen model."""
    require_worker_token(request)
    global VOICE_PROMPT
    voice_file = (req.voice_file or "").strip()
    reference_audio_file = (req.reference_audio_file or "").strip()
    validate_readable_path(voice_file, "voice_file")
    validate_readable_path(reference_audio_file, "reference_audio_file")

    started_at = time.monotonic()
    debug_log(
        "voice config update begin voice_file=%s reference_audio=%s",
        bool(voice_file),
        bool(reference_audio_file),
    )
    with LOCK:
        try:
            loaded_voice_prompt = cached_voice_prompt(voice_file) if voice_file else None
        except Exception as exc:
            if debug_enabled():
                LOGGER.exception("[QwenWorker] voice config update failed")
            raise HTTPException(status_code=400, detail=f"{type(exc).__name__}: {exc}") from exc
        VOICE_PROMPT = loaded_voice_prompt
        CONFIG["voice_file"] = voice_file
        CONFIG["reference_audio_file"] = reference_audio_file
        CONFIG["reference_text"] = req.reference_text or ""
        CONFIG["x_vector_only_mode"] = bool(req.x_vector_only_mode)

    snapshot = voice_config_snapshot()
    debug_log(
        "voice config update done voice_file=%s reference_audio=%s items=%s elapsed=%.3fs",
        snapshot["voice_file"],
        snapshot["reference_audio_file"],
        snapshot["voice_prompt_items"],
        time.monotonic() - started_at,
    )
    return {"ok": True, **snapshot}


@app.post("/synthesize")
def synthesize(req: SynthesizeRequest, request: Request) -> Response:
    require_worker_token(request)
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty.")

    request_id = uuid.uuid4().hex[:8]
    job = SynthesisJob(request_id=request_id, request=req, created_at=time.monotonic())
    accepted, queued_count, max_queue_size = try_add_queued_request(job)
    if not accepted:
        record_queue_rejection(job, queued_count, max_queue_size)
        debug_log(
            "request %s rejected queue_full text_len=%s queued_count=%s max_queue_size=%s",
            request_id,
            len(text),
            queued_count,
            max_queue_size,
        )
        raise HTTPException(
            status_code=429,
            detail=(
                "TTS queue is full "
                f"({queued_count}/{max_queue_size}). Please retry after the current tasks finish."
            ),
        )
    debug_log(
        "request %s queued text_len=%s language=%s tone=%s queued_count=%s max_queue_size=%s",
        request_id,
        len(text),
        (req.language or CONFIG.get("language") or "Auto").strip() or "Auto",
        (req.tone or "").strip(),
        queued_count,
        max_queue_size,
    )
    JOB_QUEUE.put(job)
    job.done.wait()
    if job.error:
        if isinstance(job.error, HTTPException):
            raise job.error
        raise HTTPException(
            status_code=500,
            detail=f"{type(job.error).__name__}: {job.error}",
        ) from job.error
    return Response(content=job.result or b"", media_type="audio/wav")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    global CONFIG
    with open(args.config, "r", encoding="utf-8") as f:
        CONFIG = json.load(f)

    debug = debug_enabled()
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    debug_log(
        "config loaded host=%s port=%s fingerprint=%s",
        CONFIG.get("host") or "127.0.0.1",
        int(CONFIG.get("port") or 8514),
        str(CONFIG.get("fingerprint") or "")[:12],
    )

    if CONFIG.get("hf_home"):
        os.environ["HF_HOME"] = str(CONFIG["hf_home"])
    if CONFIG.get("hf_endpoint"):
        os.environ["HF_ENDPOINT"] = str(CONFIG["hf_endpoint"])
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    uvicorn.run(
        app,
        host=CONFIG.get("host") or "127.0.0.1",
        port=int(CONFIG.get("port") or 8514),
        log_level="debug" if debug else "info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

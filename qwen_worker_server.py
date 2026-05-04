from __future__ import annotations

import argparse
import io
import json
import logging
import os
import threading
import time
import uuid
from typing import Any

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem


CONFIG: dict[str, Any] = {}
MODEL: Qwen3TTSModel | None = None
VOICE_PROMPT: list[VoiceClonePromptItem] | None = None
LOCK = threading.Lock()
ACTIVE_LOCK = threading.Lock()
ACTIVE_REQUEST: dict[str, Any] | None = None
LOGGER = logging.getLogger("qwen_local_tts_worker")
app = FastAPI(title="Qwen Local TTS Worker")


class SynthesizeRequest(BaseModel):
    text: str
    language: str = "Auto"


class VoiceConfigRequest(BaseModel):
    voice_file: str = ""
    reference_audio_file: str = ""
    reference_text: str = ""
    x_vector_only_mode: bool = False


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


def debug_enabled() -> bool:
    return bool_from_config(CONFIG.get("debug_logging"), False)


def debug_log(message: str, *args: Any) -> None:
    if debug_enabled():
        LOGGER.debug("[QwenWorker] " + message, *args)


def set_active_request(request_id: str, text_len: int, language: str, source: str) -> None:
    global ACTIVE_REQUEST
    with ACTIVE_LOCK:
        ACTIVE_REQUEST = {
            "request_id": request_id,
            "text_len": text_len,
            "language": language,
            "source": source,
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
        VOICE_PROMPT = load_voice_prompt(voice_file)
        debug_log("voice prompt loaded items=%s", len(VOICE_PROMPT))
    debug_log("startup complete elapsed=%.3fs", time.monotonic() - started_at)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": MODEL is not None,
        "model": CONFIG.get("model"),
        "debug_logging": debug_enabled(),
        "fingerprint": CONFIG.get("fingerprint", ""),
        "busy": active_request_snapshot() is not None,
        "active_request": active_request_snapshot(),
        **voice_config_snapshot(),
    }


@app.post("/voice_config")
def update_voice_config(req: VoiceConfigRequest) -> dict[str, Any]:
    """Hot-load the active voice without reloading the Qwen model."""
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
            loaded_voice_prompt = load_voice_prompt(voice_file) if voice_file else None
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
def synthesize(req: SynthesizeRequest) -> Response:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty.")

    request_id = uuid.uuid4().hex[:8]
    started_at = time.monotonic()
    language = (req.language or CONFIG.get("language") or "Auto").strip() or "Auto"
    generation = dict(CONFIG.get("generation") or {})
    source = "voice_prompt" if VOICE_PROMPT else "reference_audio"
    debug_log(
        "request %s synthesize start text_len=%s language=%s source=%s",
        request_id,
        len(text),
        language,
        source,
    )

    with LOCK:
        set_active_request(request_id, len(text), language, source)
        try:
            if VOICE_PROMPT:
                wavs, sr = MODEL.generate_voice_clone(
                    text=text,
                    language=language,
                    voice_clone_prompt=VOICE_PROMPT,
                    **generation,
                )
            else:
                ref_audio = CONFIG.get("reference_audio_file") or ""
                if not ref_audio:
                    raise ValueError("voice_file or reference_audio_file must be configured.")
                ref_text = (CONFIG.get("reference_text") or "").strip() or None
                use_xvec = bool(CONFIG.get("x_vector_only_mode", False))
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
    return Response(content=audio, media_type="audio/wav")


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

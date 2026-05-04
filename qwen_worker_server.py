from __future__ import annotations

import argparse
import io
import json
import os
import threading
from dataclasses import asdict
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
app = FastAPI(title="Qwen Local TTS Worker")


class SynthesizeRequest(BaseModel):
    text: str
    language: str = "Auto"


def dtype_from_str(value: str) -> torch.dtype:
    value = (value or "bfloat16").lower()
    if value in ("bf16", "bfloat16"):
        return torch.bfloat16
    if value in ("fp16", "float16"):
        return torch.float16
    if value in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


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
    MODEL = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map=CONFIG.get("device") or "mps",
        dtype=dtype_from_str(CONFIG.get("dtype") or "bfloat16"),
        attn_implementation=CONFIG.get("attn_implementation") or "sdpa",
    )
    voice_file = CONFIG.get("voice_file") or ""
    if voice_file:
        VOICE_PROMPT = load_voice_prompt(voice_file)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": MODEL is not None,
        "model": CONFIG.get("model"),
        "voice_file": bool(CONFIG.get("voice_file")),
        "reference_audio_file": bool(CONFIG.get("reference_audio_file")),
        "fingerprint": CONFIG.get("fingerprint", ""),
    }


@app.post("/synthesize")
def synthesize(req: SynthesizeRequest) -> Response:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty.")

    language = (req.language or CONFIG.get("language") or "Auto").strip() or "Auto"
    generation = dict(CONFIG.get("generation") or {})

    with LOCK:
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
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    return Response(content=wav_bytes(wavs[0], sr), media_type="audio/wav")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    global CONFIG
    with open(args.config, "r", encoding="utf-8") as f:
        CONFIG = json.load(f)

    if CONFIG.get("hf_home"):
        os.environ["HF_HOME"] = str(CONFIG["hf_home"])
    if CONFIG.get("hf_endpoint"):
        os.environ["HF_ENDPOINT"] = str(CONFIG["hf_endpoint"])
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    uvicorn.run(
        app,
        host=CONFIG.get("host") or "127.0.0.1",
        port=int(CONFIG.get("port") or 8514),
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

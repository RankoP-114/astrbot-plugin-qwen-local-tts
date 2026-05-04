from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import aiohttp

try:
    from astrbot.api import logger as astrbot_logger
except Exception:  # pragma: no cover - fallback for non-AstrBot helper tests.
    astrbot_logger = logging.getLogger("qwen_local_tts")

try:
    from astrbot.core.utils.astrbot_path import get_astrbot_temp_path
except Exception:  # pragma: no cover - lets the helper be unit-tested outside AstrBot.
    def get_astrbot_temp_path() -> str:
        return tempfile.gettempdir()


PLUGIN_DIR = Path(__file__).resolve().parent
WORKER_PATH = PLUGIN_DIR / "qwen_worker_server.py"


def first_file_path(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, (list, tuple)):
        for item in value:
            found = first_file_path(item)
            if found:
                return found
        return ""
    if isinstance(value, dict):
        for key in ("path", "file", "name", "url"):
            found = first_file_path(value.get(key))
            if found:
                return found
        return ""
    if isinstance(value, str):
        if value.startswith(("http://", "https://")):
            return value
        return os.path.expanduser(value)
    return ""


def _server_parts(server_url: str) -> tuple[str, int]:
    parsed = urlparse(server_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return host, port


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _bool(value: Any, default: bool = False) -> bool:
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


class QwenLocalTTSClient:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config or {})
        self.server_url = str(self.config.get("server_url") or "http://127.0.0.1:8514").rstrip("/")
        self.auto_start_server = _bool(self.config.get("auto_start_server"), True)
        self.allow_external_server_config = _bool(
            self.config.get("allow_external_server_config"), not self.auto_start_server
        )
        self.debug_logging = _bool(self.config.get("debug_logging"), False)
        self.request_timeout = _float(self.config.get("request_timeout"), 180.0)
        self.startup_timeout = _float(self.config.get("startup_timeout"), 240.0)
        self.process: asyncio.subprocess.Process | None = None
        self.config_payload = self._build_worker_config()
        self.fingerprint = self._fingerprint(self.config_payload)
        self._debug(
            "client initialized server_url=%s auto_start=%s external_config=%s fingerprint=%s",
            self.server_url,
            self.auto_start_server,
            self.allow_external_server_config,
            self.fingerprint[:12],
        )

    def _build_worker_config(self) -> dict[str, Any]:
        host, port = _server_parts(self.server_url)
        payload = {
            "host": self.config.get("host") or host,
            "port": _int(self.config.get("port"), port),
            "model": self.config.get("model") or "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            "device": self.config.get("device") or "mps",
            "dtype": self.config.get("dtype") or "bfloat16",
            "attn_implementation": self.config.get("attn_implementation") or "sdpa",
            "hf_home": self.config.get("hf_home") or "",
            "hf_endpoint": self.config.get("hf_endpoint") or "",
            "qwen_repo_dir": self.config.get("qwen_repo_dir") or "",
            "voice_file": first_file_path(self.config.get("voice_file")),
            "reference_audio_file": first_file_path(self.config.get("reference_audio_file")),
            "reference_text": self.config.get("reference_text") or "",
            "x_vector_only_mode": _bool(self.config.get("x_vector_only_mode"), False),
            "language": self.config.get("language") or "Auto",
            "debug_logging": self.debug_logging,
            "generation": {
                "max_new_tokens": _int(self.config.get("max_new_tokens"), 2048),
                "temperature": _float(self.config.get("temperature"), 0.9),
                "top_k": _int(self.config.get("top_k"), 50),
                "top_p": _float(self.config.get("top_p"), 1.0),
                "repetition_penalty": _float(self.config.get("repetition_penalty"), 1.05),
                "subtalker_dosample": True,
                "subtalker_top_k": _int(self.config.get("subtalker_top_k"), 50),
                "subtalker_top_p": _float(self.config.get("subtalker_top_p"), 1.0),
                "subtalker_temperature": _float(self.config.get("subtalker_temperature"), 0.9),
            },
        }
        return payload

    @staticmethod
    def _fingerprint(payload: dict[str, Any]) -> str:
        relevant = dict(payload)
        relevant.pop("host", None)
        relevant.pop("port", None)
        relevant.pop("debug_logging", None)
        raw = json.dumps(relevant, sort_keys=True, ensure_ascii=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _debug(self, message: str, *args: Any) -> None:
        if self.debug_logging:
            astrbot_logger.debug("[QwenLocalTTS] " + message, *args)

    async def synthesize(self, text: str, language: str | None = None) -> str:
        text = (text or "").strip()
        if not text:
            raise ValueError("TTS text is empty.")
        request_id = uuid.uuid4().hex[:8]
        started_at = time.monotonic()
        resolved_language = language or self.config_payload.get("language") or "Auto"
        self._debug(
            "request %s synthesize start text_len=%s language=%s",
            request_id,
            len(text),
            resolved_language,
        )
        await self.ensure_server()

        output_path = os.path.join(
            get_astrbot_temp_path(),
            f"qwen_local_tts_{uuid.uuid4().hex}.wav",
        )
        request = {
            "text": text,
            "language": resolved_language,
        }
        timeout = aiohttp.ClientTimeout(total=self.request_timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{self.server_url}/synthesize", json=request) as resp:
                data = await resp.read()
                if resp.status != 200:
                    detail = data.decode("utf-8", errors="replace")
                    self._debug(
                        "request %s synthesize failed status=%s detail=%s",
                        request_id,
                        resp.status,
                        detail,
                    )
                    raise RuntimeError(f"Qwen local TTS failed ({resp.status}): {detail}")
        with open(output_path, "wb") as f:
            f.write(data)
        self._debug(
            "request %s synthesize done bytes=%s output=%s elapsed=%.3fs",
            request_id,
            len(data),
            output_path,
            time.monotonic() - started_at,
        )
        return output_path

    async def ensure_server(self) -> None:
        self._debug("checking worker health url=%s", self.server_url)
        health = await self._health()
        if health:
            server_fp = health.get("fingerprint")
            self._debug(
                "worker health ok fingerprint=%s model=%s voice_file=%s reference_audio=%s",
                str(server_fp or "")[:12],
                health.get("model"),
                health.get("voice_file"),
                health.get("reference_audio_file"),
            )
            if server_fp and server_fp != self.fingerprint:
                if self.allow_external_server_config:
                    self._debug(
                        "worker fingerprint differs but external config is trusted local=%s remote=%s",
                        self.fingerprint[:12],
                        str(server_fp)[:12],
                    )
                    return
                raise RuntimeError(
                    "Qwen local TTS worker is already running with a different config. "
                    "Stop that worker or change this plugin/provider to another port."
                )
            return
        if not self.auto_start_server:
            self._debug("worker health failed and auto_start_server is disabled")
            raise RuntimeError(f"Qwen local TTS worker is not running: {self.server_url}")
        self._debug("worker not running; starting local worker")
        await self._start_server()
        deadline = time.monotonic() + self.startup_timeout
        last_error = ""
        while time.monotonic() < deadline:
            health = await self._health()
            if health and health.get("fingerprint") == self.fingerprint:
                self._debug("worker startup confirmed fingerprint=%s", self.fingerprint[:12])
                return
            if self.process and self.process.returncode is not None:
                self._debug("worker exited during startup returncode=%s", self.process.returncode)
                raise RuntimeError(
                    f"Qwen local TTS worker exited early with code {self.process.returncode}."
                )
            if health:
                last_error = "worker is healthy but config fingerprint does not match"
                self._debug(
                    "worker startup waiting: remote fingerprint mismatch local=%s remote=%s",
                    self.fingerprint[:12],
                    str(health.get("fingerprint") or "")[:12],
                )
            await asyncio.sleep(1.0)
        self._debug("worker startup timed out last_error=%s", last_error)
        raise TimeoutError(f"Timed out waiting for Qwen local TTS worker: {last_error}")

    async def _health(self) -> dict[str, Any] | None:
        timeout = aiohttp.ClientTimeout(total=3)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{self.server_url}/health") as resp:
                    if resp.status != 200:
                        self._debug("worker health returned status=%s", resp.status)
                        return None
                    return await resp.json()
        except Exception as exc:
            self._debug("worker health request failed: %s: %s", type(exc).__name__, exc)
            return None

    async def _start_server(self) -> None:
        python_bin = str(self.config.get("python_bin") or "").strip()
        if not python_bin:
            raise ValueError("python_bin is required when auto_start_server is enabled.")
        config_fd, config_path = tempfile.mkstemp(prefix="qwen_local_tts_", suffix=".json")
        with os.fdopen(config_fd, "w", encoding="utf-8") as f:
            payload = dict(self.config_payload)
            payload["fingerprint"] = self.fingerprint
            json.dump(payload, f, ensure_ascii=True)
        self._debug("worker config written path=%s fingerprint=%s", config_path, self.fingerprint[:12])

        env = os.environ.copy()
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        if self.config_payload.get("hf_home"):
            env["HF_HOME"] = str(self.config_payload["hf_home"])
        if self.config_payload.get("hf_endpoint"):
            env["HF_ENDPOINT"] = str(self.config_payload["hf_endpoint"])

        cwd = str(self.config_payload.get("qwen_repo_dir") or "") or None
        if cwd and not os.path.isdir(cwd):
            self._debug("configured qwen_repo_dir does not exist; falling back cwd=None path=%s", cwd)
            cwd = None

        self.process = await asyncio.create_subprocess_exec(
            python_bin,
            str(WORKER_PATH),
            "--config",
            config_path,
            cwd=cwd,
            env=env,
            stdout=None if self.debug_logging else asyncio.subprocess.DEVNULL,
            stderr=None if self.debug_logging else asyncio.subprocess.DEVNULL,
        )
        self._debug(
            "worker process spawned pid=%s cwd=%s log_output=%s",
            self.process.pid,
            cwd,
            "inherited" if self.debug_logging else "discarded",
        )

    async def close(self) -> None:
        if not self.process or self.process.returncode is not None:
            return
        self._debug("terminating worker pid=%s", self.process.pid)
        self.process.terminate()
        try:
            await asyncio.wait_for(self.process.wait(), timeout=10)
        except asyncio.TimeoutError:
            self._debug("worker did not terminate in time; killing pid=%s", self.process.pid)
            self.process.kill()
            await self.process.wait()
        self._debug("worker process closed returncode=%s", self.process.returncode)

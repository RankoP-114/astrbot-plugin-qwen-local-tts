from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register
import astrbot.api.message_components as Comp
from astrbot.core.provider.entities import ProviderType
from astrbot.core.provider.provider import TTSProvider
from astrbot.core.provider import register as provider_register_module

PLUGIN_DIR = Path(__file__).resolve().parent
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from qwen_local_tts_client import QwenLocalTTSClient


DEFAULT_PROVIDER_CONFIG = {
    "id": "qwen_local_tts",
    "type": "qwen_local_tts",
    "provider_type": "text_to_speech",
    "enable": False,
    "server_url": "http://host.docker.internal:8514",
    "auto_start_server": False,
    "allow_external_server_config": True,
    "debug_logging": False,
    "python_bin": "",
    "qwen_repo_dir": "",
    "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "device": "mps",
    "dtype": "bfloat16",
    "attn_implementation": "sdpa",
    "hf_home": "",
    "hf_endpoint": "https://hf-mirror.com",
    "voice_file": "",
    "reference_audio_file": "",
    "reference_text": "",
    "x_vector_only_mode": False,
    "language": "Auto",
    "host": "127.0.0.1",
    "port": 8514,
    "request_timeout": 180.0,
    "startup_timeout": 240.0,
    "max_new_tokens": 2048,
    "temperature": 0.9,
    "top_k": 50,
    "top_p": 1.0,
    "repetition_penalty": 1.05,
}


def _register_provider_adapter_idempotent(
    provider_type_name: str,
    desc: str,
    provider_type: ProviderType,
    default_config_tmpl: dict[str, Any],
    provider_display_name: str,
):
    """AstrBot does not always unregister provider adapters during plugin hot reload."""
    provider_cls_map = getattr(provider_register_module, "provider_cls_map", None)
    provider_registry = getattr(provider_register_module, "provider_registry", None)
    if isinstance(provider_cls_map, dict) and provider_type_name in provider_cls_map:
        provider_cls_map.pop(provider_type_name, None)
        if isinstance(provider_registry, list):
            provider_registry[:] = [
                meta
                for meta in provider_registry
                if getattr(meta, "type", None) != provider_type_name
            ]
        logger.debug(
            "Removed stale provider adapter before re-registering: %s",
            provider_type_name,
        )
    return provider_register_module.register_provider_adapter(
        provider_type_name,
        desc,
        provider_type=provider_type,
        default_config_tmpl=default_config_tmpl,
        provider_display_name=provider_display_name,
    )


@_register_provider_adapter_idempotent(
    "qwen_local_tts",
    "Local Qwen3-TTS voice-file provider",
    provider_type=ProviderType.TEXT_TO_SPEECH,
    default_config_tmpl=dict(DEFAULT_PROVIDER_CONFIG),
    provider_display_name="Qwen Local TTS",
)
class ProviderQwenLocalTTS(TTSProvider):
    def __init__(self, provider_config: dict[str, Any], provider_settings: dict[str, Any]) -> None:
        super().__init__(provider_config, provider_settings)
        merged = dict(DEFAULT_PROVIDER_CONFIG)
        merged.update(provider_config or {})
        self.client = QwenLocalTTSClient(merged)
        self.set_model(merged.get("model") or "Qwen/Qwen3-TTS-12Hz-1.7B-Base")

    async def get_audio(self, text: str) -> str:
        return await self.client.synthesize(text)

    async def terminate(self) -> None:
        await self.client.close()


@register(
    "astrbot_plugin_qwen_local_tts",
    "RankoP-114",
    "Generate QQ voice messages with a locally hosted Qwen3-TTS voice file.",
    "0.1.0",
)
class QwenLocalTTSPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig) -> None:
        super().__init__(context)
        self.config = config
        self.client = QwenLocalTTSClient(dict(config or {}))

    @filter.command("qwentts")
    async def qwentts(self, event: AstrMessageEvent, text: str):
        """Use the configured Qwen voice file to generate a QQ voice message."""
        text = (text or "").strip()
        if not text:
            yield event.plain_result("用法：/qwentts 要合成的文本")
            return
        try:
            audio_path = await self.client.synthesize(text, language=self.config.get("language", "Auto"))
            yield event.chain_result([Comp.Record(file=audio_path, url=audio_path)])
        except Exception as exc:
            logger.error("Qwen Local TTS failed: %s", exc)
            yield event.plain_result(f"Qwen Local TTS 生成失败：{type(exc).__name__}: {exc}")

    async def terminate(self) -> None:
        await self.client.close()

from __future__ import annotations

import sys
from pathlib import Path

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register
import astrbot.api.message_components as Comp

PLUGIN_DIR = Path(__file__).resolve().parent
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from qwen_local_tts_client import QwenLocalTTSClient


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
            audio_path = await self.client.synthesize(
                text,
                language=self.config.get("language", "Auto"),
            )
            yield event.chain_result([Comp.Record(file=audio_path, url=audio_path)])
        except Exception as exc:
            logger.error("Qwen Local TTS failed: %s", exc)
            yield event.plain_result(
                f"Qwen Local TTS 生成失败：{type(exc).__name__}: {exc}",
            )

    async def terminate(self) -> None:
        await self.client.close()

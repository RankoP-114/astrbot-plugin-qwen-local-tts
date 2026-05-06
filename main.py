# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register
import astrbot.api.message_components as Comp

PLUGIN_DIR = Path(__file__).resolve().parent
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from qwen_local_tts_client import QwenLocalTTSClient


LANGUAGE_ALIASES = {
    "auto": "Auto",
    "自动": "Auto",
    "zh": "Chinese",
    "cn": "Chinese",
    "chinese": "Chinese",
    "中文": "Chinese",
    "汉语": "Chinese",
    "普通话": "Chinese",
    "ja": "Japanese",
    "jp": "Japanese",
    "japanese": "Japanese",
    "日语": "Japanese",
    "日文": "Japanese",
    "日本语": "Japanese",
    "en": "English",
    "english": "English",
    "英语": "English",
    "英文": "English",
    "ko": "Korean",
    "korean": "Korean",
    "韩语": "Korean",
    "韩文": "Korean",
    "fr": "French",
    "french": "French",
    "法语": "French",
    "de": "German",
    "german": "German",
    "德语": "German",
    "es": "Spanish",
    "spanish": "Spanish",
    "西班牙语": "Spanish",
    "pt": "Portuguese",
    "portuguese": "Portuguese",
    "葡萄牙语": "Portuguese",
    "ru": "Russian",
    "russian": "Russian",
    "俄语": "Russian",
    "it": "Italian",
    "italian": "Italian",
    "意大利语": "Italian",
}

LANGUAGE_LABELS = {
    "Auto": "自动",
    "Chinese": "中文",
    "English": "英语",
    "Japanese": "日语",
    "Korean": "韩语",
    "French": "法语",
    "German": "德语",
    "Spanish": "西班牙语",
    "Portuguese": "葡萄牙语",
    "Russian": "俄语",
    "Italian": "意大利语",
}

TONE_ALIASES = {
    "none": "",
    "default": "",
    "auto": "",
    "无": "",
    "默认": "",
    "happy": "happy",
    "cheerful": "happy",
    "joyful": "happy",
    "开心": "happy",
    "高兴": "happy",
    "快乐": "happy",
    "愉快": "happy",
    "sad": "sad",
    "悲伤": "sad",
    "伤心": "sad",
    "难过": "sad",
    "angry": "angry",
    "mad": "angry",
    "生气": "angry",
    "愤怒": "angry",
    "gentle": "gentle",
    "soft": "gentle",
    "tender": "gentle",
    "温柔": "gentle",
    "轻柔": "gentle",
    "calm": "calm",
    "peaceful": "calm",
    "平静": "calm",
    "冷静": "calm",
    "excited": "excited",
    "energetic": "excited",
    "元气": "excited",
    "兴奋": "excited",
    "激动": "excited",
    "shy": "shy",
    "cute": "shy",
    "害羞": "shy",
    "撒娇": "shy",
}

TONE_LABELS = {
    "happy": "开心",
    "sad": "悲伤",
    "angry": "生气",
    "gentle": "温柔",
    "calm": "平静",
    "excited": "兴奋",
    "shy": "害羞",
}

TONE_REPLY_RULES = {
    "happy": "回复语气要开心、轻快，但不要夸张。",
    "sad": "回复语气要低落、柔和，但不要写哭腔说明。",
    "angry": "回复语气要有点生气、直接，但不要攻击用户。",
    "gentle": "回复语气要温柔、亲切。",
    "calm": "回复语气要平静、稳重。",
    "excited": "回复语气要兴奋、有活力，但不要过度使用感叹号。",
    "shy": "回复语气要害羞、可爱，但不要写括号动作。",
}

LANGUAGE_PATTERN = "|".join(
    re.escape(item)
    for item in sorted(LANGUAGE_ALIASES, key=len, reverse=True)
)
TONE_PATTERN = "|".join(
    re.escape(item)
    for item in sorted((item for item in TONE_ALIASES if TONE_ALIASES[item]), key=len, reverse=True)
)
LANGUAGE_OPTION_RE = re.compile(
    r"\[(?:lang|language|语言)\s*=\s*([^\]]+)\]",
    re.IGNORECASE,
)
TONE_OPTION_RE = re.compile(
    r"\[(?:tone|style|emotion|语气|情绪|风格)\s*=\s*([^\]]+)\]",
    re.IGNORECASE,
)
LANGUAGE_PREFIX_RE = re.compile(
    r"^(?:lang|language|语言)\s*=\s*([^\s]+)\s*",
    re.IGNORECASE,
)
TONE_PREFIX_RE = re.compile(
    r"^(?:tone|style|emotion|语气|情绪|风格)\s*=\s*([^\s]+)\s*",
    re.IGNORECASE,
)
QWENTTS_COMMAND_RE = re.compile(
    r"(?:^|\s)/?qwentts(?:\s+|$)(?P<text>.*)$",
    re.IGNORECASE | re.DOTALL,
)
LEADING_LANGUAGE_TONE_REQUEST_RE = re.compile(
    rf"^(?:请)?用(?P<lang>{LANGUAGE_PATTERN})(?P<tone>{TONE_PATTERN})(?:的|地)?"
    r"(?:语气|口吻|情绪|风格)?(?:语音)?"
    r"(?P<verb>回答|回复|说|讲)(?:一下)?[：:，,\s]*(?P<prompt>.+)$",
    re.IGNORECASE,
)
LEADING_TONE_LANGUAGE_REQUEST_RE = re.compile(
    rf"^(?:请)?用(?P<tone>{TONE_PATTERN})(?:的|地)?(?P<lang>{LANGUAGE_PATTERN})"
    r"(?:语音)?(?P<verb>回答|回复|说|讲)(?:一下)?[：:，,\s]*(?P<prompt>.+)$",
    re.IGNORECASE,
)
LEADING_TONE_REQUEST_RE = re.compile(
    rf"^(?:请)?用(?P<tone>{TONE_PATTERN})(?:的|地)?(?:语气|口吻|情绪|风格)?(?:语音)?"
    r"(?P<verb>回答|回复|说|讲)(?:一下)?[：:，,\s]*(?P<prompt>.+)$",
    re.IGNORECASE,
)
LEADING_LANGUAGE_REQUEST_RE = re.compile(
    rf"^(?:请)?用(?P<lang>{LANGUAGE_PATTERN})(?:语音)?"
    r"(?P<verb>回答|回复|说|讲)(?:一下)?[：:，,\s]*(?P<prompt>.+)$",
    re.IGNORECASE,
)
LEADING_VOICE_REQUEST_RE = re.compile(
    r"^(?:请)?(?:用)?语音(?P<verb>回答|回复|说|讲)(?:一下)?"
    r"[：:，,\s]*(?P<prompt>.+)$",
    re.IGNORECASE,
)
TRAILING_VOICE_REQUEST_RE = re.compile(
    rf"^(?P<prompt>.+?)[，,。；;\s]+(?:请)?用(?P<lang>{LANGUAGE_PATTERN})?"
    rf"(?P<tone>{TONE_PATTERN})?(?:的|地)?(?:语气|口吻|情绪|风格)?"
    r"(?:语音)?(?P<verb>回答|回复|说|讲)(?:一下)?[。.!！\s]*$",
    re.IGNORECASE,
)


def _config_bool(config: AstrBotConfig, key: str, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on", "enable", "enabled"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", "disable", "disabled"}:
            return False
    return bool(value)


def _config_int(config: AstrBotConfig, key: str, default: int) -> int:
    try:
        return int(config.get(key, default))
    except (TypeError, ValueError):
        return default


def _normalize_language(value: Any) -> str | None:
    text = str(value or "").strip().strip("[]").strip()
    if not text:
        return None
    return LANGUAGE_ALIASES.get(text.lower())


def _normalize_tone(value: Any) -> str | None:
    text = str(value or "").strip().strip("[]").strip()
    if not text:
        return None
    normalized = re.sub(r"[\s_-]+", "", text.lower())
    tone = TONE_ALIASES.get(normalized)
    if tone == "":
        return None
    return tone


def _detect_language_from_text(text: str) -> str:
    if re.search(r"[\u3040-\u30ff]", text):
        return "Japanese"
    if re.search(r"[\uac00-\ud7af]", text):
        return "Korean"
    if re.search(r"[\u4e00-\u9fff]", text):
        return "Chinese"
    if re.search(r"[A-Za-z]", text):
        return "English"
    return "Auto"


def _squash_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


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
    async def qwentts(self, event: AstrMessageEvent, text: str = ""):
        """Use the configured Qwen voice file to generate a QQ voice message."""
        if self._requires_at_but_missing(event):
            self._debug("skip /qwentts in group because bot was not mentioned")
            return

        event.stop_event()
        text = self._qwentts_command_text(event, text)
        language_override, tone_override, text = self._parse_tts_options(text)
        language = self._resolve_language(text, language_override)
        tone = self._resolve_tone(tone_override)
        if not text:
            yield event.plain_result("用法：/qwentts [lang=chinese] [tone=happy] 要合成的文本")
            return
        try:
            self._refresh_client_config()
            audio_path = await self.client.synthesize(text, language=language, tone=tone)
            yield event.chain_result([Comp.Record(file=audio_path, url=audio_path)])
        except Exception as exc:
            logger.error("Qwen Local TTS failed: %r", exc, exc_info=True)
            yield event.plain_result(self._format_tts_error(exc))

    @filter.on_llm_request(priority=1919811)
    async def on_llm_request(self, event: AstrMessageEvent, req):
        """Mark natural voice-reply requests in AstrBot's normal LLM path."""
        if event.get_extra("qwen_local_tts_voice_reply", False):
            return
        if self._requires_at_but_missing(event):
            return

        message = self._message_text(event)
        if not message or message.startswith("/"):
            return

        parsed = self._parse_voice_reply_request(message)
        if not parsed:
            return

        language_override, tone_override, prompt = parsed
        language = self._resolve_language(prompt, language_override)
        tone = self._resolve_tone(tone_override)
        if not prompt:
            return

        event.set_extra("qwen_local_tts_voice_reply", True)
        event.set_extra("qwen_local_tts_language", language)
        event.set_extra("qwen_local_tts_tone", tone or "")
        event.set_extra("qwen_local_tts_source_prompt_len", len(prompt))

        self._debug(
            "voice reply LLM request matched prompt_len=%s language=%s tone=%s",
            len(prompt),
            language,
            tone or "",
        )
        if req is not None:
            req.prompt = self._build_voice_reply_prompt(prompt, language, tone)
            voice_system_prompt = self._build_voice_reply_system_prompt(language, tone)
            req.system_prompt = (
                ((getattr(req, "system_prompt", "") or "").rstrip() + "\n\n" + voice_system_prompt)
                .strip()
            )

    @filter.on_decorating_result(priority=90000)
    async def on_decorating_result(self, event: AstrMessageEvent):
        """Replace the marked LLM text result with a QQ voice message."""
        if not event.get_extra("qwen_local_tts_voice_reply", False):
            return
        if event.get_extra("qwen_local_tts_voice_done", False):
            return

        result = event.get_result()
        if not result or not getattr(result, "chain", None):
            return

        content_type = str(getattr(result, "result_content_type", ""))
        if content_type.endswith("STREAMING_RESULT"):
            return

        text = _squash_spaces(result.get_plain_text())
        if not text:
            return

        event.set_extra("qwen_local_tts_voice_done", True)
        max_chars = max(20, _config_int(self.config, "voice_reply_max_chars", 220))
        if len(text) > max_chars:
            self._debug(
                "voice reply LLM text truncated text_len=%s max_chars=%s",
                len(text),
                max_chars,
            )
            text = text[:max_chars].rstrip() + "..."

        language = str(event.get_extra("qwen_local_tts_language", "Auto") or "Auto")
        language = self._resolve_language(text, language)
        tone = str(event.get_extra("qwen_local_tts_tone", "") or "") or None
        try:
            self._refresh_client_config()
            audio_path = await self.client.synthesize(text, language=language, tone=tone)
            result.chain = [Comp.Record(file=audio_path, url=audio_path)]
            result.use_t2i_ = False
            result.use_markdown_ = False
        except Exception as exc:
            event.set_extra("qwen_local_tts_voice_failed", True)
            logger.error("Qwen Local TTS voice reply failed: %r", exc, exc_info=True)
            result.chain.append(Comp.Plain("\n" + self._format_tts_error(exc)))

    async def terminate(self) -> None:
        await self.client.close()

    def _debug(self, message: str, *args: Any) -> None:
        if _config_bool(self.config, "debug_logging", False):
            logger.debug("[QwenLocalTTS] " + message, *args)

    def _refresh_client_config(self) -> None:
        self.client.refresh_config(dict(self.config or {}))

    def _message_text(self, event: AstrMessageEvent) -> str:
        try:
            return (event.get_message_str() or "").strip()
        except Exception:
            return (getattr(event, "message_str", "") or "").strip()

    def _qwentts_command_text(self, event: AstrMessageEvent, text_arg: str = "") -> str:
        raw = self._message_text(event)
        match = QWENTTS_COMMAND_RE.search(raw)
        if match:
            return match.group("text").strip()
        return (text_arg or "").strip()

    def _is_group_message(self, event: AstrMessageEvent) -> bool:
        try:
            return bool(event.get_group_id())
        except Exception:
            return False

    def _mentions_self(self, event: AstrMessageEvent) -> bool:
        self_id = ""
        try:
            self_id = str(event.get_self_id() or "").strip()
        except Exception:
            self_id = ""
        if not self_id:
            return bool(getattr(event, "is_at_or_wake_command", False))

        try:
            messages = event.get_messages()
        except Exception:
            messages = getattr(getattr(event, "message_obj", None), "message", []) or []
        for comp in messages:
            if isinstance(comp, Comp.At) and str(comp.qq) == self_id:
                return True
        return False

    def _requires_at_but_missing(self, event: AstrMessageEvent) -> bool:
        if not _config_bool(self.config, "require_at_in_group", True):
            return False
        return self._is_group_message(event) and not self._mentions_self(event)

    def _default_language(self) -> str:
        return _normalize_language(self.config.get("language", "Auto")) or "Auto"

    def _default_tone(self) -> str | None:
        return _normalize_tone(self.config.get("default_tone", ""))

    def _resolve_language(self, text: str, override: str | None = None) -> str:
        if override:
            return override
        language = self._default_language()
        if language == "Auto" and _config_bool(self.config, "auto_detect_language", True):
            return _detect_language_from_text(text)
        return language

    def _resolve_tone(self, override: str | None = None) -> str | None:
        return override or self._default_tone()

    def _parse_tts_options(self, text: str) -> tuple[str | None, str | None, str]:
        remaining = (text or "").strip()
        language: str | None = None
        tone: str | None = None

        match = LANGUAGE_OPTION_RE.search(remaining)
        if match:
            language = _normalize_language(match.group(1)) or language
            remaining = (remaining[: match.start()] + remaining[match.end() :]).strip()

        match = TONE_OPTION_RE.search(remaining)
        if match:
            tone = _normalize_tone(match.group(1)) or tone
            remaining = (remaining[: match.start()] + remaining[match.end() :]).strip()

        for _ in range(4):
            match = LANGUAGE_PREFIX_RE.match(remaining)
            if match:
                language = _normalize_language(match.group(1)) or language
                remaining = remaining[match.end() :].strip()
                continue
            match = TONE_PREFIX_RE.match(remaining)
            if match:
                tone = _normalize_tone(match.group(1)) or tone
                remaining = remaining[match.end() :].strip()
                continue
            break

        natural = self._match_natural_voice_request(remaining)
        if natural:
            natural_language, natural_tone, prompt = natural
            language = natural_language or language
            tone = natural_tone or tone
            remaining = prompt

        return language, tone, remaining

    def _parse_voice_reply_request(self, text: str) -> tuple[str | None, str | None, str] | None:
        normalized = (text or "").strip()
        if not normalized:
            return None

        natural = self._match_natural_voice_request(normalized)
        if natural:
            return natural

        match = LEADING_VOICE_REQUEST_RE.match(normalized)
        if match:
            return None, None, match.group("prompt").strip()

        match = TRAILING_VOICE_REQUEST_RE.match(normalized)
        if match:
            return (
                _normalize_language(match.group("lang")),
                _normalize_tone(match.group("tone")),
                match.group("prompt").strip(),
            )

        return None

    def _match_natural_voice_request(self, text: str) -> tuple[str | None, str | None, str] | None:
        normalized = (text or "").strip()
        for pattern in (
            LEADING_LANGUAGE_TONE_REQUEST_RE,
            LEADING_TONE_LANGUAGE_REQUEST_RE,
            LEADING_TONE_REQUEST_RE,
            LEADING_LANGUAGE_REQUEST_RE,
        ):
            match = pattern.match(normalized)
            if match:
                return (
                    _normalize_language(match.groupdict().get("lang")),
                    _normalize_tone(match.groupdict().get("tone")),
                    match.group("prompt").strip(),
                )
        return None

    def _build_voice_reply_system_prompt(self, language: str, tone: str | None = None) -> str:
        language_label = LANGUAGE_LABELS.get(language, "用户要求的语言")
        if language == "Auto":
            language_rule = "使用用户消息中最自然的语言回答。"
        else:
            language_rule = f"必须使用{language_label}回答。"
        tone_rule = TONE_REPLY_RULES.get(tone or "", "")
        return (
            "你正在为语音消息生成回复。"
            f"{language_rule}"
            f"{tone_rule}"
            "只输出适合直接朗读的回复正文，不要写标题、列表、Markdown、括号动作或语音合成说明。"
            "回复要自然、简短。"
        )

    def _build_voice_reply_prompt(self, prompt: str, language: str, tone: str | None = None) -> str:
        max_chars = max(20, _config_int(self.config, "voice_reply_max_chars", 220))
        language_label = LANGUAGE_LABELS.get(language, "自然语言")
        tone_label = TONE_LABELS.get(tone or "")
        tone_clause = f"，语气偏{tone_label}" if tone_label else ""
        if language == "Auto":
            return f"{prompt}\n\n请用适合直接朗读的方式简短回答{tone_clause}，控制在 {max_chars} 个字符以内。"
        return f"{prompt}\n\n请用{language_label}简短回答{tone_clause}，控制在 {max_chars} 个字符以内。"

    def _format_tts_error(self, exc: Exception) -> str:
        if isinstance(exc, TimeoutError):
            timeout = getattr(self.client, "request_timeout", None)
            suffix = f"（当前等待 {timeout:g} 秒）" if isinstance(timeout, (int, float)) else ""
            return (
                "Qwen Local TTS 生成超时"
                f"{suffix}：Worker 可能仍在生成过长音频。"
                "建议指定语言重试，例如 /qwentts [lang=chinese] 你好。"
            )
        detail = str(exc).strip()
        if detail:
            return f"Qwen Local TTS 生成失败：{type(exc).__name__}: {detail}"
        return f"Qwen Local TTS 生成失败：{type(exc).__name__}"

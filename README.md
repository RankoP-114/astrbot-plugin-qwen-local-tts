# astrbot-plugin-qwen-local-tts

**Language / 语言**: [简体中文](#简体中文) | [English](#english)

<a id="简体中文"></a>

## 简体中文

本插件让 AstrBot 使用本机运行的 Qwen3-TTS 生成 QQ 语音消息。

### 功能

- 提供 `/qwentts <文本>` 指令，可在 QQ 中直接生成语音消息。
- 支持 `/qwentts [lang=chinese] <文本>` 临时指定合成语言。
- 支持“用语音回答”“用中文说”“用日语说”等自然语言触发：先调用 AstrBot 当前 LLM 生成回复，再把回复转成 QQ 语音。
- 群聊默认必须 @ 机器人后才会触发，避免误把普通群聊内容转成语音。
- 支持使用 Qwen3-TTS WebUI 保存出的 `voice_clone_prompt_*.pt` 音色文件。
- 支持 AstrBot 后台更换音色后自动热同步到宿主机 Worker，无需每次重启 Worker。
- 支持为中文、日语、英语分别配置不同 `.pt` 音色文件。
- 推荐部署方式：AstrBot 在 Docker 中运行，Qwen3-TTS 在宿主机本地运行，Docker 只通过 HTTP 调用。
- 提供可开关的 Debug 日志，便于排查 Docker 连通性、Worker 启动和语音生成错误。

### 架构

```text
QQ / OneBot
    |
 AstrBot in Docker
    |
 http://host.docker.internal:8514
    |
 Qwen Worker on host
    |
 Qwen3-TTS model + voice file
```

在这种模式下，Docker 容器里不需要安装 Qwen3-TTS、PyTorch 或模型文件。容器只安装本插件和 `aiohttp`，把文本发送到宿主机 Worker，Worker 返回 WAV 音频。

### 音色文件

推荐使用 Qwen3-TTS WebUI 保存出的 `.pt` 音色文件：

1. 打开 Qwen3-TTS Base WebUI。
2. 进入 `Save / Load Voice`。
3. 上传参考音频，并填写对应的准确文本。
4. 点击 `Save Voice File`。
5. 在 AstrBot 插件后台的“Qwen 音色文件”中上传或选择保存出的 `voice_clone_prompt_*.pt`。

也可以直接配置 `reference_audio_file` 和 `reference_text` 作为备用方案，但每次生成时都需要处理参考音频，速度通常不如 `.pt` 音色文件。

如果 AstrBot 在 Docker 中运行、Qwen Worker 在宿主机运行，请同时填写“宿主机插件数据目录”，让插件能把后台的 `files/voice_file/*.pt` 相对路径转换成宿主机绝对路径。

插件后台可以单独配置：

- `Qwen 音色文件`：通用兜底音色。
- `中文专用音色文件`：语言为 `Chinese` 或“用中文说”时优先使用。
- `日语专用音色文件`：语言为 `Japanese` 或“用日语说/用日文说”时优先使用。
- `英语专用音色文件`：语言为 `English` 或“用英语说/用英文说”时优先使用。

专用音色留空时，会自动回退到通用音色。

### 部署：AstrBot 在 Docker，Qwen 在本机

#### 1. 在宿主机准备 Qwen3-TTS

先确保宿主机上已经能运行 Qwen3-TTS，并且能用 Qwen3-TTS 的 Python 环境导入 `qwen_tts`。

```bash
<path-to-qwen-venv-python> -c "from qwen_tts import Qwen3TTSModel; print('ok')"
```

macOS Apple Silicon 通常建议使用：

```text
device = mps
attn_implementation = sdpa
PYTORCH_ENABLE_MPS_FALLBACK = 1
```

#### 2. 创建宿主机 Worker 配置

复制示例配置：

```bash
cd <plugin-dir>
cp host_worker_config.example.json host_worker_config.local.json
```

编辑 `host_worker_config.local.json`，至少确认这些字段：

```json
{
  "host": "0.0.0.0",
  "port": 8514,
  "voice_file": "",
  "device": "mps",
  "dtype": "bfloat16",
  "attn_implementation": "sdpa",
  "debug_logging": false
}
```

说明：

- `host` 要保持 `0.0.0.0`，这样 Docker 容器才能访问宿主机 Worker。
- `voice_file` 可以留空；如果开启后台音色同步，插件会在生成前通过 `/voice_config` 把当前音色热加载到 Worker。
- `hf_home` 可以留空，也可以设置为宿主机上的 Hugging Face 缓存目录。
- `fingerprint` 可保持示例值；Docker 模式下插件配置会信任外部 Worker。

#### 3. 启动宿主机 Worker

在宿主机运行：

```bash
cd <plugin-dir>
export QWEN_TTS_PYTHON="<path-to-qwen-venv-python>"
export QWEN_TTS_REPO="<path-to-qwen3-tts-repo>"
./start_host_worker.sh ./host_worker_config.local.json
```

启动后检查：

```bash
curl http://127.0.0.1:8514/health
```

如果 AstrBot Docker 运行在 macOS 或 Windows Docker Desktop 中，容器内访问宿主机通常使用：

```text
http://host.docker.internal:8514
```

如果是 Linux Docker，可能需要在 `docker-compose.yml` 中加入：

```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

#### 4. 安装插件到 AstrBot

将本插件放到 AstrBot 插件目录：

```text
AstrBot/data/plugins/astrbot-plugin-qwen-local-tts
```

在 AstrBot 容器内安装依赖：

```bash
pip install -r data/plugins/astrbot-plugin-qwen-local-tts/requirements.txt
```

然后重启 AstrBot。

#### 5. 配置 AstrBot 后台

插件配置中填写：

```text
server_url = http://host.docker.internal:8514
auto_start_server = false
allow_external_server_config = true
sync_voice_to_external_worker = true
host_plugin_data_dir = /path/to/AstrBot/data/plugin_data/astrbot_plugin_qwen_local_tts
debug_logging = false
require_at_in_group = true
auto_detect_language = true
voice_reply_max_chars = 220
```

其中 `host_plugin_data_dir` 要填写宿主机上的真实目录，不是 Docker 容器里的路径。这个目录下面应能看到 `files/voice_file/voice_clone_prompt_*.pt`。

本插件不注册 AstrBot Provider，QQ 中直接使用 `/qwentts` 指令。

### QQ 使用

群聊中默认需要先 @ 机器人。

直接把文本转成语音：

发送：

```text
/qwentts 你好，这是本地 Qwen3-TTS 生成的语音。
```

插件会向宿主机 Qwen Worker 请求合成，并发送 QQ 语音消息。

临时指定语言：

```text
/qwentts [lang=chinese] 你好，我是茉莉。
/qwentts [lang=japanese] おはようございます。
/qwentts [lang=english] Good morning.
```

自然语言触发会先让 AstrBot 当前 LLM 回答，再把 LLM 的回答转成语音：

```text
用语音回答 介绍一下你自己
用中文说 讲个很短的早安
用日语说 夸我一句
用日文说 介绍一下你自己
用英语说 介绍一下你自己
```

`lang` 支持 `auto`、`chinese`、`english`、`japanese`、`korean`、`french`、`german`、`spanish`、`portuguese`、`russian`、`italian`，也支持 `中文`、`日语`、`日文`、`英语` 等中文写法。

注意：QQ 官方机器人接口可能不支持语音消息。推荐使用 QQ 个人号 / OneBot v11 平台，例如 NapCat 或 aiocqhttp。

### 常见问题

#### Docker 里访问不到 Worker

确认宿主机 Worker 监听的是 `0.0.0.0:8514`，不是 `127.0.0.1:8514`。

在 AstrBot 容器里测试：

```bash
curl http://host.docker.internal:8514/health
```

Linux Docker 如果无法解析 `host.docker.internal`，给 compose 加上：

```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

#### 生成失败：找不到音色文件

如果开启了“同步后台音色到外部 Worker”，请确认：

- AstrBot 后台的“Qwen 音色文件”已经选择了 `voice_clone_prompt_*.pt`。
- “宿主机插件数据目录”填写的是宿主机真实目录，例如 `/path/to/AstrBot/data/plugin_data/astrbot_plugin_qwen_local_tts`。
- 该目录下能找到 `files/voice_file/voice_clone_prompt_*.pt`。

如果不开启同步，则 `voice_file` 必须写在宿主机 Worker 配置中，并且路径必须是宿主机真实存在的绝对路径。不要把宿主机音色路径填到 Docker 容器内的插件配置里。

#### 后台换音色不生效

开启 `sync_voice_to_external_worker` 后，插件会在每次生成前调用宿主机 Worker 的 `/voice_config` 接口热加载当前音色。旧版本 Worker 不支持该接口，或 `host_plugin_data_dir` 未填写时，Worker 可能仍然使用启动时的旧音色。

#### 私聊和群聊音色听起来不一样

如果后台刚换过音色，最常见原因是 AstrBot 后台配置已更新，但宿主机 Worker 仍在使用旧的 `voice_file`。开启音色同步并确认 Worker 已更新后，私聊和群聊会使用同一个 Worker 音色。

仍有轻微差异时，通常不是不同音色文件，而是以下因素造成的：私聊和群聊给 LLM 的上下文不同，LLM 返回文本长短和语气不同；Qwen3-TTS 的采样也会让相同音色在不同文本上产生不同的语速、情绪和音高。

#### 首次生成很慢

首次启动 Worker 会加载模型，首次生成也可能触发模型缓存读取或 MPS 编译。之后会复用同一个 Worker 进程。

#### 生成超时：TimeoutError

如果 AstrBot 提示 `Qwen Local TTS 生成超时`，通常是 Worker 仍在生成过长音频，或者 Qwen3-TTS 的 `Auto` 语言判断偶发生成了异常长的音频。

建议：

- 优先指定语言，例如 `/qwentts [lang=chinese] 你好`。
- 保持 `auto_detect_language = true`，插件会在默认 `Auto` 时按文本自动改用 `Chinese`、`Japanese`、`Korean` 或 `English`。
- 自然语言触发会受到 `voice_reply_max_chars` 限制，避免 LLM 回复太长。
- 如果 Worker 长时间占用模型，可重启宿主机 Worker 后重试。

#### 麦克风或 WebUI 和插件无关

本插件不依赖 Qwen3-TTS 的 Gradio WebUI。WebUI 只用于保存音色文件；真正给 AstrBot 用的是 `qwen_worker_server.py`。

#### 如何打开 Debug 日志

如果要排查问题：

- 在 AstrBot 插件配置中将 `debug_logging` 改为 `true`，用于打印 AstrBot 侧的健康检查、请求开始/结束、耗时和错误信息。
- 在宿主机 `host_worker_config.local.json` 中将 `debug_logging` 改为 `true`，用于打印 Qwen Worker 侧的模型加载、音色加载、生成耗时和异常堆栈。
- Debug 日志只记录文本长度，不记录完整待合成文本。

[返回顶部](#astrbot-plugin-qwen-local-tts)

<a id="english"></a>

## English

This plugin lets AstrBot generate QQ voice messages with a Qwen3-TTS instance running on the host machine.

### Features

- Provides `/qwentts <text>` for direct QQ voice-message generation.
- Supports `/qwentts [lang=chinese] <text>` for per-message language override.
- Supports natural triggers such as "用语音回答", "用中文说", and "用日语说": the plugin first asks AstrBot's current LLM, then turns the LLM reply into a QQ voice message.
- Group chats require mentioning the bot by default, so ordinary group messages do not trigger TTS accidentally.
- Supports `voice_clone_prompt_*.pt` voice files saved from the Qwen3-TTS WebUI.
- Automatically hot-syncs the voice selected in AstrBot settings to the host worker before synthesis.
- Supports separate `.pt` voice files for Chinese, Japanese, and English.
- Recommended deployment: run AstrBot in Docker, run Qwen3-TTS on the host, and let Docker call it over HTTP.
- Provides switchable debug logs for diagnosing Docker connectivity, worker startup, and synthesis errors.

### Architecture

```text
QQ / OneBot
    |
 AstrBot in Docker
    |
 http://host.docker.internal:8514
    |
 Qwen Worker on host
    |
 Qwen3-TTS model + voice file
```

In this mode, the Docker container does not need Qwen3-TTS, PyTorch, or model files. It only installs this plugin and `aiohttp`, sends text to the host worker, and receives WAV audio back.

### Voice File

It is recommended to use the `.pt` voice file saved by the Qwen3-TTS WebUI:

1. Open the Qwen3-TTS Base WebUI.
2. Go to `Save / Load Voice`.
3. Upload the reference audio and fill in the exact transcript.
4. Click `Save Voice File`.
5. Upload or select the saved `voice_clone_prompt_*.pt` in the AstrBot plugin settings.

You can also configure `reference_audio_file` and `reference_text` as a fallback, but that requires processing the reference audio during generation and is usually slower than using the `.pt` voice file.

If AstrBot runs in Docker while the Qwen worker runs on the host, also set the host plugin data directory so the plugin can translate `files/voice_file/*.pt` from AstrBot settings into a real host path.

The plugin settings can hold separate voices:

- `Qwen 音色文件`: the default fallback voice.
- `中文专用音色文件`: used first when the language is `Chinese` or the message says "用中文说".
- `日语专用音色文件`: used first when the language is `Japanese` or the message says "用日语说/用日文说".
- `英语专用音色文件`: used first when the language is `English` or the message says "用英语说/用英文说".

If a language-specific voice is empty, the plugin falls back to the default voice.

### Deployment: AstrBot in Docker, Qwen on Host

#### 1. Prepare Qwen3-TTS on the Host

Make sure Qwen3-TTS already works on the host and its Python environment can import `qwen_tts`.

```bash
<path-to-qwen-venv-python> -c "from qwen_tts import Qwen3TTSModel; print('ok')"
```

For Apple Silicon macOS, the usual settings are:

```text
device = mps
attn_implementation = sdpa
PYTORCH_ENABLE_MPS_FALLBACK = 1
```

#### 2. Create the Host Worker Config

Copy the example config:

```bash
cd <plugin-dir>
cp host_worker_config.example.json host_worker_config.local.json
```

Edit `host_worker_config.local.json` and at least check these fields:

```json
{
  "host": "0.0.0.0",
  "port": 8514,
  "voice_file": "",
  "device": "mps",
  "dtype": "bfloat16",
  "attn_implementation": "sdpa",
  "debug_logging": false
}
```

Notes:

- Keep `host` as `0.0.0.0` so the Docker container can reach the host worker.
- `voice_file` can be empty; when voice sync is enabled, the plugin calls `/voice_config` before synthesis and hot-loads the current voice into the worker.
- `hf_home` can be empty, or point to a Hugging Face cache directory on the host.
- `fingerprint` can keep the example value; Docker mode trusts the external worker config.

#### 3. Start the Host Worker

Run this on the host:

```bash
cd <plugin-dir>
export QWEN_TTS_PYTHON="<path-to-qwen-venv-python>"
export QWEN_TTS_REPO="<path-to-qwen3-tts-repo>"
./start_host_worker.sh ./host_worker_config.local.json
```

Check after startup:

```bash
curl http://127.0.0.1:8514/health
```

If AstrBot Docker runs on Docker Desktop for macOS or Windows, the container usually reaches the host through:

```text
http://host.docker.internal:8514
```

For Linux Docker, you may need this in `docker-compose.yml`:

```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

#### 4. Install the Plugin into AstrBot

Place this plugin under AstrBot's plugin directory:

```text
AstrBot/data/plugins/astrbot-plugin-qwen-local-tts
```

Install dependencies inside the AstrBot container:

```bash
pip install -r data/plugins/astrbot-plugin-qwen-local-tts/requirements.txt
```

Then restart AstrBot.

#### 5. Configure AstrBot

Set these values in the plugin config:

```text
server_url = http://host.docker.internal:8514
auto_start_server = false
allow_external_server_config = true
sync_voice_to_external_worker = true
host_plugin_data_dir = /path/to/AstrBot/data/plugin_data/astrbot_plugin_qwen_local_tts
debug_logging = false
require_at_in_group = true
auto_detect_language = true
voice_reply_max_chars = 220
```

`host_plugin_data_dir` must be the real directory on the host, not a path inside the Docker container. It should contain `files/voice_file/voice_clone_prompt_*.pt`.

This plugin does not register an AstrBot provider. Use the `/qwentts` command directly in QQ.

### QQ Usage

In group chats, mention the bot first by default.

Direct text-to-speech:

Send:

```text
/qwentts Hello, this voice message is generated by local Qwen3-TTS.
```

The plugin sends the text to the host Qwen worker and returns a QQ voice message.

Override the language for one message:

```text
/qwentts [lang=chinese] 你好，我是茉莉。
/qwentts [lang=japanese] おはようございます。
/qwentts [lang=english] Good morning.
```

Natural triggers ask AstrBot's current LLM first, then synthesize the LLM reply:

```text
用语音回答 介绍一下你自己
用中文说 讲个很短的早安
用日语说 夸我一句
用日文说 介绍一下你自己
用英语说 介绍一下你自己
```

`lang` supports `auto`, `chinese`, `english`, `japanese`, `korean`, `french`, `german`, `spanish`, `portuguese`, `russian`, and `italian`, plus Chinese aliases such as `中文`, `日语`, `日文`, and `英语`.

Note: QQ official bot APIs may not support voice messages. QQ personal-account / OneBot v11 platforms such as NapCat or aiocqhttp are the intended targets.

### Troubleshooting

#### Docker Cannot Reach the Worker

Make sure the host worker binds to `0.0.0.0:8514`, not `127.0.0.1:8514`.

Test inside the AstrBot container:

```bash
curl http://host.docker.internal:8514/health
```

If Linux Docker cannot resolve `host.docker.internal`, add this to compose:

```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

#### Voice File Not Found

If voice sync is enabled, check that:

- The AstrBot plugin setting selects a `voice_clone_prompt_*.pt` file.
- `host_plugin_data_dir` points to the real host directory, for example `/path/to/AstrBot/data/plugin_data/astrbot_plugin_qwen_local_tts`.
- The file exists under `files/voice_file/voice_clone_prompt_*.pt` inside that directory.

If voice sync is disabled, `voice_file` must be set in the host worker config, and it must be a real absolute path on the host. Do not put the host voice path into the plugin config inside Docker.

#### Voice Changes Do Not Take Effect

With `sync_voice_to_external_worker` enabled, the plugin calls the host worker's `/voice_config` endpoint before each synthesis request. Older workers do not support this endpoint, and an empty `host_plugin_data_dir` can leave the worker using the voice file it loaded at startup.

#### Private and Group Chats Sound Different

Right after changing the voice, the most common cause is that AstrBot settings changed but the host worker was still using its old `voice_file`. Once voice sync is enabled and the worker has updated, private and group chats use the same worker voice.

Small remaining differences are usually not different voice files. Private and group chats can send different context to the LLM, so the generated text, length, tone, and punctuation may differ; Qwen3-TTS sampling can also change speed, emotion, and pitch slightly across different text.

#### First Generation Is Slow

The first worker startup loads the model, and the first generation may trigger cache reads or MPS compilation. Later requests reuse the same worker process.

#### TimeoutError During Generation

If AstrBot reports `Qwen Local TTS 生成超时`, the worker is usually still generating an unusually long audio file, or Qwen3-TTS `Auto` language detection produced an overlong result.

Recommended fixes:

- Prefer an explicit language, for example `/qwentts [lang=chinese] 你好`.
- Keep `auto_detect_language = true`; when the default language is `Auto`, the plugin will infer `Chinese`, `Japanese`, `Korean`, or `English` from the text.
- Natural LLM voice replies are limited by `voice_reply_max_chars` to avoid very long speech.
- If the worker is stuck for a long time, restart the host worker and try again.

#### WebUI Microphone Is Unrelated

This plugin does not depend on the Qwen3-TTS Gradio WebUI. The WebUI is only used to save the voice file; AstrBot talks to `qwen_worker_server.py`.

#### How to Enable Debug Logs

When troubleshooting:

- Set `debug_logging` to `true` in the AstrBot plugin config to print AstrBot-side health checks, request start/end events, elapsed time, and error details.
- Set `debug_logging` to `true` in the host `host_worker_config.local.json` to print Qwen worker-side model loading, voice loading, synthesis timing, and exception stack traces.
- Debug logs record text length only. They do not record the full synthesis text.

[Back to top](#astrbot-plugin-qwen-local-tts)

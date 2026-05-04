# astrbot-plugin-qwen-local-tts

**Language / 语言**: [简体中文](#简体中文) | [English](#english)

<a id="简体中文"></a>

## 简体中文

本插件让 AstrBot 使用本机运行的 Qwen3-TTS 生成 QQ 语音消息。

### 功能

- 注册 `qwen_local_tts` TTS Provider，可接入 AstrBot 的标准 TTS 流程。
- 提供 `/qwentts <文本>` 指令，可在 QQ 中直接测试语音生成。
- 支持使用 Qwen3-TTS WebUI 保存出的 `voice_clone_prompt_*.pt` 音色文件。
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
5. 将保存出的 `voice_clone_prompt_*.pt` 路径填入宿主机 Worker 配置里的 `voice_file`。

也可以直接配置 `reference_audio_file` 和 `reference_text` 作为备用方案，但每次生成时都需要处理参考音频，速度通常不如 `.pt` 音色文件。

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
  "voice_file": "/absolute/path/to/voice_clone_prompt_xxx.pt",
  "device": "mps",
  "dtype": "bfloat16",
  "attn_implementation": "sdpa",
  "debug_logging": false
}
```

说明：

- `host` 要保持 `0.0.0.0`，这样 Docker 容器才能访问宿主机 Worker。
- `voice_file` 是宿主机上的文件路径，不是 Docker 容器里的路径。
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

插件配置或 Provider 配置中填写：

```text
server_url = http://host.docker.internal:8514
auto_start_server = false
allow_external_server_config = true
debug_logging = false
```

如果要接入 AstrBot 的自动 TTS 流程，添加或启用 Provider：

```text
type = qwen_local_tts
id = qwen_local_tts
```

如果只想在 QQ 中手动测试，可以直接使用 `/qwentts` 指令。

### QQ 使用

发送：

```text
/qwentts 你好，这是本地 Qwen3-TTS 生成的语音。
```

插件会向宿主机 Qwen Worker 请求合成，并发送 QQ 语音消息。

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

`voice_file` 必须写在宿主机 Worker 配置中，并且路径必须是宿主机真实存在的绝对路径。不要把宿主机音色路径填到 Docker 容器内的插件配置里。

#### 首次生成很慢

首次启动 Worker 会加载模型，首次生成也可能触发模型缓存读取或 MPS 编译。之后会复用同一个 Worker 进程。

#### 麦克风或 WebUI 和插件无关

本插件不依赖 Qwen3-TTS 的 Gradio WebUI。WebUI 只用于保存音色文件；真正给 AstrBot 用的是 `qwen_worker_server.py`。

#### 如何打开 Debug 日志

如果要排查问题：

- 在 AstrBot 插件配置或 Provider 配置中将 `debug_logging` 改为 `true`，用于打印 AstrBot 侧的健康检查、请求开始/结束、耗时和错误信息。
- 在宿主机 `host_worker_config.local.json` 中将 `debug_logging` 改为 `true`，用于打印 Qwen Worker 侧的模型加载、音色加载、生成耗时和异常堆栈。
- Debug 日志只记录文本长度，不记录完整待合成文本。

[返回顶部](#astrbot-plugin-qwen-local-tts)

<a id="english"></a>

## English

This plugin lets AstrBot generate QQ voice messages with a Qwen3-TTS instance running on the host machine.

### Features

- Registers a `qwen_local_tts` TTS provider for AstrBot's normal TTS pipeline.
- Provides `/qwentts <text>` for direct QQ voice-message testing.
- Supports `voice_clone_prompt_*.pt` voice files saved from the Qwen3-TTS WebUI.
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
5. Put the saved `voice_clone_prompt_*.pt` path into `voice_file` in the host worker config.

You can also configure `reference_audio_file` and `reference_text` as a fallback, but that requires processing the reference audio during generation and is usually slower than using the `.pt` voice file.

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
  "voice_file": "/absolute/path/to/voice_clone_prompt_xxx.pt",
  "device": "mps",
  "dtype": "bfloat16",
  "attn_implementation": "sdpa",
  "debug_logging": false
}
```

Notes:

- Keep `host` as `0.0.0.0` so the Docker container can reach the host worker.
- `voice_file` is a host filesystem path, not a path inside the Docker container.
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

Set these values in the plugin config or provider config:

```text
server_url = http://host.docker.internal:8514
auto_start_server = false
allow_external_server_config = true
debug_logging = false
```

To use AstrBot's automatic TTS pipeline, add or enable this provider:

```text
type = qwen_local_tts
id = qwen_local_tts
```

If you only want to test manually in QQ, use the `/qwentts` command directly.

### QQ Usage

Send:

```text
/qwentts Hello, this voice message is generated by local Qwen3-TTS.
```

The plugin sends the text to the host Qwen worker and returns a QQ voice message.

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

`voice_file` must be set in the host worker config, and it must be a real absolute path on the host. Do not put the host voice path into the plugin config inside Docker.

#### First Generation Is Slow

The first worker startup loads the model, and the first generation may trigger cache reads or MPS compilation. Later requests reuse the same worker process.

#### WebUI Microphone Is Unrelated

This plugin does not depend on the Qwen3-TTS Gradio WebUI. The WebUI is only used to save the voice file; AstrBot talks to `qwen_worker_server.py`.

#### How to Enable Debug Logs

When troubleshooting:

- Set `debug_logging` to `true` in the AstrBot plugin config or provider config to print AstrBot-side health checks, request start/end events, elapsed time, and error details.
- Set `debug_logging` to `true` in the host `host_worker_config.local.json` to print Qwen worker-side model loading, voice loading, synthesis timing, and exception stack traces.
- Debug logs record text length only. They do not record the full synthesis text.

[Back to top](#astrbot-plugin-qwen-local-tts)

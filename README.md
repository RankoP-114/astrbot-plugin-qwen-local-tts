# astrbot-plugin-qwen-local-tts

Local Qwen3-TTS support for AstrBot.

This plugin provides two entry points:

- A `qwen_local_tts` AstrBot TTS provider for AstrBot's normal TTS pipeline.
- A `/qwentts <text>` command that sends a QQ voice message using the voice file configured in the plugin settings.

## What voice file means

Use the `voice_clone_prompt_*.pt` file produced by the Qwen3-TTS WebUI:

1. Open the Qwen3-TTS Base WebUI.
2. Go to `Save / Load Voice`.
3. Upload a reference audio file and fill in the exact transcript.
4. Click `Save Voice File`.
5. Upload that `.pt` file in this plugin's backend setting `voice_file`.

You can also configure `reference_audio_file` and `reference_text` as a fallback, but the saved `.pt` voice file is faster because the voice prompt is precomputed.

## Local worker

The plugin can start `qwen_worker_server.py` with a configured Qwen3-TTS Python environment, but the recommended Docker setup is to run the worker on the host and let AstrBot call it over HTTP.

- Python: set `QWEN_TTS_PYTHON` or configure `python_bin`.
- Qwen repo: set `QWEN_TTS_REPO` or configure `qwen_repo_dir`.
- Worker URL from Docker: `http://host.docker.internal:8514`

The model is loaded once in the worker process and reused for later requests.

## AstrBot in Docker, Qwen on macOS host

Yes. Keep Qwen3-TTS on the macOS host and let AstrBot Docker call it over HTTP.

1. Copy `host_worker_config.example.json` to your own config file and set `voice_file` to the host path of the saved Qwen `.pt` voice file.
2. Start the worker on macOS:

```bash
cd <plugin-dir>
export QWEN_TTS_PYTHON="<path-to-qwen-venv-python>"
export QWEN_TTS_REPO="<path-to-qwen3-tts-repo>"
./start_host_worker.sh ./host_worker_config.example.json
```

3. In AstrBot Docker plugin/provider settings:

```text
server_url = http://host.docker.internal:8514
auto_start_server = false
allow_external_server_config = true
```

In this mode Docker does not install or load Qwen3-TTS. It only sends text to the macOS host worker and receives a WAV file back.

## Use in QQ

For a quick QQ test, send:

```text
/qwentts 你好，这是本地 Qwen3-TTS 生成的语音。
```

For automatic AstrBot reply-to-voice output:

1. Install this plugin under `AstrBot/data/plugins/astrbot-plugin-qwen-local-tts`.
2. Install plugin dependencies from `requirements.txt`.
3. Add a TTS provider of type `qwen_local_tts`.
4. Set `voice_file` to the saved Qwen `.pt` voice file path.
5. Enable AstrBot TTS and select provider id `qwen_local_tts`.

QQ official bot interfaces may not support voice records. QQ personal account / OneBot v11 (`aiocqhttp`, for example NapCat) is the intended target.

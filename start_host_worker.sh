#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later

set -euo pipefail

PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-$PLUGIN_DIR/host_worker_config.example.json}"
PYTHON_BIN="${QWEN_TTS_PYTHON:-python3}"
QWEN_REPO_DIR="${QWEN_TTS_REPO:-}"

if [[ "$CONFIG_PATH" != /* ]]; then
  CONFIG_PATH="$(pwd)/$CONFIG_PATH"
fi

if [[ -n "$QWEN_REPO_DIR" ]]; then
  cd "$QWEN_REPO_DIR"
fi
exec "$PYTHON_BIN" "$PLUGIN_DIR/qwen_worker_server.py" --config "$CONFIG_PATH"

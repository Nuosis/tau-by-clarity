#!/usr/bin/env bash
# tau-by-clarity container entrypoint.
#
# 1. Start ollama serve in the background (if not already up).
# 2. Wait for /api/tags to respond (max 30s).
# 3. Exec the requested tau command (default: `tau`).
#
# Tau auto-degrades to deterministic embeddings if Ollama is unreachable, so
# this script never blocks the user from running; it just maximizes the
# chance of having a real embedding service ready.

set -e

OLLAMA_PORT="${OLLAMA_PORT:-11434}"
OLLAMA_LOOPBACK="127.0.0.1"

start_ollama() {
  if curl -sf "http://${OLLAMA_LOOPBACK}:${OLLAMA_PORT}/api/tags" >/dev/null 2>&1; then
    return 0
  fi
  if ! command -v ollama >/dev/null 2>&1; then
    echo "[entrypoint] ollama not found on PATH; tau will use deterministic embeddings" >&2
    return 0
  fi
  echo "[entrypoint] starting ollama serve on port ${OLLAMA_PORT}..." >&2
  ollama serve >/tmp/ollama.log 2>&1 &
  OLLAMA_BG_PID=$!
  for i in $(seq 1 30); do
    if curl -sf "http://${OLLAMA_LOOPBACK}:${OLLAMA_PORT}/api/tags" >/dev/null 2>&1; then
      echo "[entrypoint] ollama ready" >&2
      return 0
    fi
    if ! kill -0 "${OLLAMA_BG_PID}" 2>/dev/null; then
      echo "[entrypoint] ollama serve exited prematurely; see /tmp/ollama.log" >&2
      return 0
    fi
    sleep 1
  done
  echo "[entrypoint] ollama did not become ready within 30s; tau will degrade" >&2
  return 0
}

start_ollama
exec "$@"

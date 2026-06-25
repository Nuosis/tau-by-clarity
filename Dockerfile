# Tau-by-clarity container image — bundles the Python agent with a local Ollama
# service for memory embeddings. Pre-pulls the default embed model so the
# first session is fully usable.
#
# Build:   docker build -t tau:dev .
# Run:     docker run -it --rm -e ANTHROPIC_API_KEY=... tau:dev
#          docker run -it --rm -e ANTHROPIC_API_KEY=... tau:dev --print "hello"
#          docker run -it --rm -e ANTHROPIC_API_KEY=... tau:dev --setup-ollama
#
# Architecture: single stage, slim base. Ollama runs as a background process
# started by the entrypoint, tau runs as the foreground process.

FROM python:3.13-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:${PATH}" \
    OLLAMA_HOST="LOOPBACK_PII:11434" \
    OLLAMA_PORT=11434

# System deps: curl for the Ollama install script, ca-certificates for HTTPS.
# build-essential is intentionally omitted; tau's deps are pure-Python wheels.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Install Ollama (Linux one-shot installer; pinned to a known-good version).
# https://ollama.ai/download — adjust OLLAMA_VERSION to upgrade.
ARG OLLAMA_VERSION=0.5.7
RUN curl -fsSL https://ollama.ai/install.sh | sh \
 && ollama --version

# Pre-pull the default embedding model. This bakes ~270MB into the image so
# the first session doesn't pay the download cost. Override with
# --build-arg TAU_EMBED_MODEL=... for a different default.
ARG TAU_EMBED_MODEL=nomic-embed-text
RUN ollama serve & \
    OLLAMA_PID=$! \
 && for i in $(seq 1 30); do \
        curl -sf "http://LOOPBACK_PII:11434/api/tags" >/dev/null && break; \
        sleep 1; \
    done \
 && ollama pull "${TAU_EMBED_MODEL}" \
 && kill "${OLLAMA_PID}" \
 && wait "${OLLAMA_PID}" 2>/dev/null || true

# Install tau-by-clarity from this source tree. The wheel build below also
# happens at image build time, so `pip install /src` re-uses the built wheel.
WORKDIR /src
COPY pyproject.toml uv.lock ./
COPY packages/ ./packages/
COPY skills/ ./skills/
COPY README.md LICENSE PARITY.md parity_audit.py eval_parity.py ./
RUN pip install --no-cache-dir .

# Working dirs the runtime expects: agent dir for sessions / ccr.db, project
# dir for memory.db. The entrypoint mounts /work as the working project.
RUN mkdir -p /root/.tau/agent /work
WORKDIR /work

# Container entrypoint: ensure ollama serve is up, then exec tau. Tau
# auto-degrades to deterministic embeddings if Ollama is unreachable, so
# this also covers the "no model pulled yet" case gracefully.
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["tau"]

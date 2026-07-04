FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml .
# --extra-index-url: torch/torchvision are pinned to the +cu121 local version, which
# only exists on download.pytorch.org (PyPI's default build targets a newer CUDA that
# the deploy VM's driver can't run) — see the pyproject.toml comment.
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --system -e '.[dev]' \
    --extra-index-url https://download.pytorch.org/whl/cu121
COPY . .
EXPOSE 8000
CMD ["granian","--interface","asgi","--host","0.0.0.0","--port","8000","--workers","1","--runtime-threads","2","--http","auto","--respawn-failed-workers","main:app"]

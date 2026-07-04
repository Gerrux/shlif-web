FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml .
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --system -e '.[dev]'
COPY . .
EXPOSE 8000
CMD ["granian","--interface","asgi","--host","0.0.0.0","--port","8000","--workers","1","--runtime-threads","2","--http","auto","--respawn-failed-workers","main:app"]

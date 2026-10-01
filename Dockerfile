# Cloud Run image. Build the matrix first so it ships inside the image.
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app.py ./
COPY twenty_q ./twenty_q
COPY static ./static

ENV PORT=8080
CMD ["sh", "-c", "uv run --no-sync uvicorn app:app --host 0.0.0.0 --port $PORT"]

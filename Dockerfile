FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

ENV HOST=0.0.0.0
ENV PORT=8000
ENV PYTHONUTF8=1
EXPOSE 8000

CMD ["sh", "-c", "uv run --no-sync uvicorn app:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000}"]

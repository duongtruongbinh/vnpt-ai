FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

# Runtime stage
FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY --from=builder /app/.venv /app/.venv

COPY src/ ./src/
COPY app.py ./
COPY pyproject.toml ./

# Copy crawled data and documents into container
COPY data/ /crawled_data/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV DATA_INPUT_DIR=/data
ENV DATA_OUTPUT_DIR=/output
ENV KB_DATA_DIR=/crawled_data
ENV PYTHONPATH=/app

# Create directories
RUN mkdir -p /data /output /crawled_data

# Copy entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "app.py"]
FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

# Install Python 3.12 and basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /code

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application code
COPY src/ ./src/
COPY predict.py ./
COPY pyproject.toml ./

# Copy data folder (knowledge base data)
COPY data/ ./data/

# Set environment variables
ENV PATH="/code/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV DATA_INPUT_DIR=/data
ENV DATA_OUTPUT_DIR=/output
ENV PYTHONPATH=/code

# Create directories for input/output
RUN mkdir -p /data /output

# Copy entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "predict.py"]
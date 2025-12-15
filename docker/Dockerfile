FROM python:3.12-slim

# Install basic utilities and git
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Install dependencies
COPY requirements.txt ./
RUN pip3 install --no-cache-dir --upgrade pip && \ 
    pip3 install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY predict.py ./
COPY scripts/inference.sh ./inference.sh
RUN chmod +x inference.sh

# Copy knowledge base data
COPY data/qdrant_storage/ ./data/qdrant_storage/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/code
ENV DATA_INPUT_DIR=/code
ENV DATA_OUTPUT_DIR=/code
ENV INPUT_FILE_PATH="/code/private_test.json"
ENV OUTPUT_FILE_PATH="/code/submission.csv"
ENV VECTOR_DB_PATH=/code/data/qdrant_storage

# Entrypoint
CMD ["bash", "inference.sh"]
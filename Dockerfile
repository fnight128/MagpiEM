FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    libffi-dev \
    # tidy up
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt pyproject.toml setup.py ./

RUN pip install --no-cache-dir --upgrade pip setuptools wheel numpy

COPY magpiem/ ./magpiem/
COPY MANIFEST.in ./

RUN pip install --no-cache-dir .

FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash magpiem

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

RUN mkdir -p cache past_cleaning_params && \
    chown -R magpiem:magpiem /app

ENV DOCKER_CONTAINER=true

USER magpiem

EXPOSE 8050

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8050/ || exit 1

CMD ["magpiem", "--no-browser"]




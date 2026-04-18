# Slim RunPod-ready Docker image. Works on CPU or any CUDA-enabled GPU the
# endpoint runs on. The Chronos-2 model is ~120 M params so cold-start on
# CPU is ~15 s, on a small GPU (T4 / A10) it's ~3 s.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SURGE_DATA_DIR=/workspace/data

RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY runpod ./runpod

RUN pip install -e ".[api]" runpod

# Optional: pre-download the model at build time so cold starts don't fetch
# it. Set SURGE_MODEL_HF_ID at build time to bake it in.
ARG SURGE_MODEL_HF_ID=surge-grid/chronos2-full-v2
RUN python -c "from chronos import BaseChronosPipeline; BaseChronosPipeline.from_pretrained('${SURGE_MODEL_HF_ID}')"

CMD ["python", "-u", "runpod/handler.py"]

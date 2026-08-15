# Lean CPU/ONNX runtime image — the fine-tuned model is fetched from Hugging
# Face Hub at container startup (ADAS_MODEL_REPO / ADAS_MODEL_FILE below),
# not baked into the image, so the image and the model version independently.
# ONNX Runtime on CPU alone matches our benchmarked GPU throughput (~45 FPS
# at imgsz=960 — see docs/PLAN.md), so no CUDA base image is needed here.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ src/

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -e ".[serve]"

ENV ADAS_MODEL_REPO=mokshhere/adas-kitti-yolo11m
ENV ADAS_MODEL_FILE=best.onnx

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "adas.api:app", "--host", "0.0.0.0", "--port", "8000"]

# Deployed layout (see scripts/publish_space.py, which assembles this from
# the main repo: pyproject.toml + src/adas/ from the repo root, app.py +
# samples/ from space/). Not directly buildable from the main repo's root —
# this Dockerfile expects the flattened Space-repo layout as its context.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ src/

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -e ".[serve]" gradio

COPY app.py ./
COPY samples/ ./samples/

ENV ADAS_MODEL_REPO=mokshhere/adas-kitti-yolo11m

EXPOSE 7860

CMD ["python", "app.py"]

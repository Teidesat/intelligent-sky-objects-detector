# Base con CUDA 12.6 y cuDNN 9 (compatible con RTX 5060)
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY pyproject.toml .
COPY extract_deps.py .

RUN pip install --no-cache-dir tomli
RUN python extract_deps.py
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r dev-requirements.txt || true

# PyTorch con soporte CUDA 12.6
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu126

RUN mkdir -p /app/src /app/data /app/docs

CMD ["python", "-m", "model_training.train"]
# Base image: TensorFlow official with GPU + CUDA support
FROM tensorflow/tensorflow:2.16.1-gpu

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only pyproject.toml first (for layer caching)
COPY pyproject.toml .

# Install tomli (to read pyproject.toml in Python 3.10)
RUN pip install --no-cache-dir tomli

# Extract and install production dependencies (excluding tensorflow if desired)
RUN python -c "import tomli; data = tomli.load(open('pyproject.toml', 'rb')); deps = data['project']['dependencies']; print('\n'.join(deps))" > requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# Optionally install development dependencies
RUN python -c "import tomli; data = tomli.load(open('pyproject.toml', 'rb')); dev_deps = data['project']['optional-dependencies'].get('dev', []); print('\n'.join(dev_deps))" > dev-requirements.txt && \
    pip install --no-cache-dir -r dev-requirements.txt || true

# Create necessary directories (optional, as volumes will be mounted)
RUN mkdir -p /app/src /app/data /app/docs

# (Optional) Create a non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Default command (can be overridden)
CMD ["python", "src/model_training/train.py"]
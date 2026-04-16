FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG TEST_ENV

WORKDIR /app

RUN apt-get -y update && apt-get install -y \
    git wget g++ gcc \
    libsm6 libxext6 libffi-dev python3-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_CACHE_DIR=/.cache \
    PORT=9090 \
    WORKERS=1 \
    THREADS=4 \
    CUDA_HOME=/opt/conda \
    TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX;8.9;9.0"

# Install base requirements (gunicorn + label-studio-ml)
COPY requirements-base.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements-base.txt

# Install additional requirements (opencv, python-dotenv)
COPY requirements.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements.txt

# Install SAM2 from official Meta source (pip install, no local clone needed)
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install git+https://github.com/facebookresearch/sam2.git

# Copy service files
COPY . ./

# Download SAM2.1 Tiny checkpoint (~150 MB)
RUN python download_models.py --model tiny

RUN chmod +x start.sh

CMD ["/app/start.sh"]

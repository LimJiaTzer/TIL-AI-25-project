# Dockerfile for building the RL image.
# The base image. Using NVIDIA CUDA base for GPU support
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu20.04

# Configures settings for the image.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.12
ENV VENV_PATH="/opt/venv_py${PYTHON_VERSION}"
ENV PATH="${VENV_PATH}/bin:$PATH"

WORKDIR /workspace

# Install Python and system dependencies
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update -o Acquire::Retries=3 \
                   -o Acquire::AllowInsecureRepositories=true \
                   -o Acquire::AllowDowngradeToInsecureRepositories=true && \
    apt-get install -y --no-install-recommends \
        -o Acquire::AllowInsecureRepositories=true \
        -o Acquire::AllowDowngradeToInsecureRepositories=true \
        gnupg \
        ca-certificates \
        wget \
        ubuntu-keyring && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update -o Acquire::Retries=3 && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update -o Acquire::Retries=3 && \
    apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        build-essential \
        cmake \
        pkg-config && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment and upgrade pip
RUN python${PYTHON_VERSION} -m venv ${VENV_PATH}
RUN ${VENV_PATH}/bin/pip install --upgrade pip

# Install PyTorch 2.7.0 with CUDA 12.8 support
RUN ${VENV_PATH}/bin/pip install \
    torch==2.7.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# Install your dependencies
COPY requirements.txt .
# Pin NumPy version before installing other requirements to avoid conflicts
RUN ${VENV_PATH}/bin/pip install "numpy<2.0"
RUN ${VENV_PATH}/bin/pip install --no-cache-dir -r requirements.txt

# Copy your source files
COPY src .

# Set runtime environment
ENV PYTHONPATH="/workspace"

# Start your model server using the venv python
CMD ["/opt/venv_py3.12/bin/python", "-m", "uvicorn", "rl_server:app", "--host", "0.0.0.0", "--port", "5004"]
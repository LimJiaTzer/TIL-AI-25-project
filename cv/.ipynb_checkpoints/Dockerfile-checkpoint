# Use an NVIDIA base image with CUDA 12.8 development toolkit
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu20.04

# Set ENV variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.12
ENV VENV_PATH="/opt/venv"
ENV PATH="${VENV_PATH}/bin:$PATH"

WORKDIR /workspace

# Install Python 3.12 and system dependencies
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
    apt-get install -y --no-install-recommends \
        software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update -o Acquire::Retries=3 && \
    apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        build-essential \
        cmake \
        pkg-config \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libgomp1 && \
    # Set python3.12 as the default python3
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} && \
    # Clean up apt lists to reduce image size
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python${PYTHON_VERSION} -m venv ${VENV_PATH}

# Upgrade pip within the venv
RUN ${VENV_PATH}/bin/pip install --upgrade pip

# Install PyTorch 2.7.0 with CUDA 12.8 support
RUN ${VENV_PATH}/bin/pip install \
    torch==2.7.0+cu128 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# Copy requirements.txt first
COPY requirements.txt .

# Pin NumPy version before installing other requirements to avoid conflicts
RUN ${VENV_PATH}/bin/pip install "numpy<2.0"

# Install Python dependencies using the venv's pip
RUN ${VENV_PATH}/bin/pip install --no-cache-dir -r requirements.txt

# Copy your application source files
COPY src/ /workspace/src/ 

# Copy your fine-tuned YOLO model
COPY models/runs/testing/best.pt /models/best_yolo.pt

# Copy the entrypoint script into the container
COPY entrypoint.sh /workspace/entrypoint.sh

# Make the entrypoint script executable
RUN chmod +x /workspace/entrypoint.sh

# Set runtime environment
ENV PYTHONPATH="/workspace"

# Set the entrypoint to our script
ENTRYPOINT ["/workspace/entrypoint.sh"]

# Expose port
EXPOSE 5002

# Command to run your application
CMD ["/opt/venv/bin/python", "-m", "uvicorn", "src.cv_server:app", "--host", "0.0.0.0", "--port", "5002"]
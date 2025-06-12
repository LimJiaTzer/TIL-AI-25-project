# Use an NVIDIA base image with CUDA 12.8 devel (already correct for cu128)
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu20.04

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.10
ENV VENV_PATH="/opt/venv_py${PYTHON_VERSION}"
ENV PATH="${VENV_PATH}/bin:$PATH"

WORKDIR /workspace

# Install Python and OS dependencies
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    # Initial update: allow insecure to fetch lists if GPG is broken for Ubuntu repos
    apt-get update -o Acquire::Retries=3 \
                   -o Acquire::AllowInsecureRepositories=true \
                   -o Acquire::AllowDowngradeToInsecureRepositories=true && \
    # Install/reinstall essential tools for GPG, certificates, and Ubuntu keyrings.
    # Also allow insecure for this step, in case GPG system is too broken to verify these critical packages for Ubuntu.
    # This is a temporary measure to bootstrap the GPG tools for Ubuntu.
    # NVIDIA keys/repos should be pre-configured by the base image.
    apt-get install -y --no-install-recommends \
        -o Acquire::AllowInsecureRepositories=true \
        -o Acquire::AllowDowngradeToInsecureRepositories=true \
        gnupg \
        ca-certificates \
        wget \
        ubuntu-keyring && \
    # Clean archives and lists immediately after installing keyring and GPG tools.
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    # Now update again. GPG tools and keyrings for Ubuntu should be fixed/updated.
    # NVIDIA repos should already be fine from the base image. This update MUST be secure.
    apt-get update -o Acquire::Retries=3 && \
    # Install software-properties-common for add-apt-repository
    apt-get install -y --no-install-recommends software-properties-common && \
    # Add the PPA for deadsnakes. The -y flag confirms automatically.
    add-apt-repository -y ppa:deadsnakes/ppa && \
    # Update package lists again to include packages from the new PPA.
    apt-get update -o Acquire::Retries=3 && \
    # Install all required application packages
    apt-get install -y --no-install-recommends \
      python${PYTHON_VERSION} \
      python${PYTHON_VERSION}-dev \
      python${PYTHON_VERSION}-venv \
      libjpeg-dev libpng-dev libtiff5-dev libwebp-dev zlib1g-dev libfreetype6-dev liblcms2-dev \
      cmake pkg-config build-essential \
      libgl1-mesa-glx libglib2.0-0 && \
    # Set up Python alternatives
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    # Final clean up to reduce layer size
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python${PYTHON_VERSION} -m venv ${VENV_PATH}
RUN ${VENV_PATH}/bin/pip install --upgrade pip

# Install PyTorch (DocTR will use this)
# Updated to PyTorch 2.7.0 for CUDA 12.8
# We also include torchvision and torchaudio and let pip resolve their versions
# based on the torch version and the CUDA index.
RUN ${VENV_PATH}/bin/pip install \
    torch==2.7.0+cu128 \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu128

# Copy new requirements.txt for DocTR
COPY requirements.txt .
# Pin NumPy version BEFORE installing requirements to avoid conflicts
# PyTorch 2.2+ (and thus 2.7.0) generally requires numpy < 2.0 for stability
RUN ${VENV_PATH}/bin/pip install "numpy<2.0"
RUN ${VENV_PATH}/bin/pip install --no-cache-dir -r requirements.txt

# --- Application Code ---
COPY src/ /workspace/src/

# --- Warmup and Model Caching for DocTR (Online during this step) ---
RUN HF_HUB_OFFLINE=0 ${VENV_PATH}/bin/python -c "\
print('Starting DocTR warmup script...'); \
import sys; \
sys.path.insert(0, '/workspace/src'); \
sys.path.insert(0, '/workspace'); \
print(f'PYTHONPATH for warmup: {sys.path}'); \
from src.ocr_manager import OCRManager; \
print('Imported OCRManager from src.ocr_manager'); \
manager = OCRManager(warmup_on_init=True); \
print('DocTR Warmup complete, models should be cached.') \
"

# --- Set Offline mode for runtime ---
ENV HF_HUB_OFFLINE="1"

# Set PYTHONPATH for the runtime CMD.
ENV PYTHONPATH="/workspace/src"

EXPOSE 5003

# Use the hardcoded expanded path for the venv Python in CMD
CMD ["/opt/venv_py3.10/bin/python", "-m", "uvicorn", "src.ocr_server:app", "--host", "0.0.0.0", "--port", "5003"]
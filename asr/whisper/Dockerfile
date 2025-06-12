FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu20.04

# --- Environment Variables ---
# Set non-interactive frontend for package installers
ENV DEBIAN_FRONTEND=noninteractive
# Python-related environment variables for cleaner execution in Docker
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Tell pip to not complain about running as root
ENV PIP_ROOT_USER_ACTION=ignore
# For easier CUDA debugging in development. Can be removed for production.
ENV CUDA_LAUNCH_BLOCKING=1
# Define the path to the virtual environment
ENV VENV_PATH=/opt/venv
# Add the virtual environment's bin directory to the system's PATH
ENV PATH="${VENV_PATH}/bin:${PATH}"
# Explicitly define CUDA_HOME and set the default C/C++ compilers
ENV CUDA_HOME=/usr/local/cuda
ENV CC=gcc-11
ENV CXX=g++-11

# Set the working directory
WORKDIR /workspace

# --- System Dependencies Installation ---
# This layer installs essential build tools, adds secure PPAs for Python 3.12 and GCC-11,
# installs them, and configures them as the system defaults.
RUN set -e; \
    echo "Starting PPA and package installation process..." && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg \
        dirmngr \
        software-properties-common \
        cmake \
        ninja-build \
        # OS-level audio libs for torchaudio/nemo
        sox libsox-dev libsox-fmt-all libsndfile1 && \
    \
    echo "Initializing GnuPG directory for root..." && \
    mkdir -p /root/.gnupg && chmod 700 /root/.gnupg && \
    (gpg-connect-agent /bye || true) && \
    \
    echo "Adding deadsnakes PPA for Python 3.12..." && \
    mkdir -p /etc/apt/keyrings && \
    gpg --no-default-keyring --keyring /etc/apt/keyrings/deadsnakes-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys F23C5A6CF475977595C89F51BA6932366A755776 && \
    echo "deb [signed-by=/etc/apt/keyrings/deadsnakes-archive-keyring.gpg] http://ppa.launchpad.net/deadsnakes/ppa/ubuntu focal main" > /etc/apt/sources.list.d/deadsnakes.list && \
    \
    echo "Adding ubuntu-toolchain-r/test PPA for GCC 11..." && \
    gpg --no-default-keyring --keyring /etc/apt/keyrings/ubuntu-toolchain-r-test-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 60C317803A41BA51845E371A1E9377A2BA9EF27F && \
    echo "deb [signed-by=/etc/apt/keyrings/ubuntu-toolchain-r-test-archive-keyring.gpg] http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu focal main" > /etc/apt/sources.list.d/ubuntu-toolchain-r-test.list && \
    \
    echo "Updating apt lists and installing Python 3.12 & GCC 11..." && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        gcc-11 \
        g++-11 && \
    \
    echo "Verifying Python.h installation..." && \
    test -f /usr/include/python3.12/Python.h || (echo "Python.h not found!" && exit 1) && \
    \
    echo "Configuring system alternatives..." && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 60 && \
    update-alternatives --set python3 /usr/bin/python3.12 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 60 && \
    update-alternatives --set gcc /usr/bin/gcc-11 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 60 && \
    update-alternatives --set g++ /usr/bin/g++-11 && \
    \
    echo "Cleaning up apt cache..." && \
    rm -rf /var/lib/apt/lists/* && \
    echo "PPA and package installation process completed."

# Verify installed versions (optional, for debugging during build)
RUN echo "Verifying versions post-installation:" && \
    python3 --version && \
    gcc --version | head -n1 && \
    g++ --version | head -n1

# --- Python Virtual Environment & Dependencies ---

# Create the virtual environment
RUN python3 -m venv ${VENV_PATH}

# Upgrade pip and setuptools
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Install PyTorch, TorchVision, and TorchAudio using the user-provided command.
RUN ${VENV_PATH}/bin/pip install --no-cache-dir \
    torch==2.7.0+cu128 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# Install bitsandbytes after PyTorch to ensure it compiles against the correct CUDA version.
RUN pip install --no-cache-dir bitsandbytes

# Copy and install the remaining Python packages from your requirements file.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The line below can be uncommented if you need to install flash-attn.
#RUN pip install flash-attn --no-build-isolation

# Copy your application source code into the image.
COPY src .

# Copy your model files into the image.
COPY Models/my_model /models/my_model

# Expose the port that the application server will listen on.
EXPOSE 5001

# Specify the command to run when the container starts.
CMD ["uvicorn", "asr_server:app", "--port", "5001", "--host", "0.0.0.0"]





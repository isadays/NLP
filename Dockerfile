# Use TensorFlow's official GPU-enabled image based on Debian 11 as the base
FROM tensorflow/tensorflow:2.12.0-gpu-debian11

# Set environment variables
ENV USER=your_username
ENV UID=1000
ENV PATH="/usr/local/bin:${PATH}"
ENV JUPYTER_KERNEL=your_jupyter_kernel_configuration

# Switch to root to install system dependencies
USER root

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    git \
    build-essential \
    apt-utils \
    && rm -rf /var/lib/apt/lists/*

# Install NVIDIA Container Toolkit (if not already installed in the base image)
# Typically, the base image already includes necessary CUDA libraries
# Uncomment if needed
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-debian11.pin \
#     && mv cuda-debian11.pin /etc/apt/preferences.d/cuda-repository-pin-600 \
#     && wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-debian11-11-8-local_11.8.0-520.61.05-1_amd64.deb \
#     && dpkg -i cuda-repo-debian11-11-8-local_11.8.0-520.61.05-1_amd64.deb \
#     && cp /var/cuda-repo-debian11-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/ \
#     && apt-get update \
#     && apt-get install -y cuda \
#     && rm cuda-repo-debian11-11-8-local_11.8.0-520.61.05-1_amd64.deb

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

# Set the working directory
WORKDIR /app

# Copy your application code into the image
COPY . /app

# Define the entry point
ENTRYPOINT ["python", "train_bert.py"]

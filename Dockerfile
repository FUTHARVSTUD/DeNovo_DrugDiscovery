# 1. Base image with CUDA and cuDNN
FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu20.04

# 2. Install OS-level prerequisites
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-pip python3-dev git && \
    rm -rf /var/lib/apt/lists/*

# 3. Set up the working directory
WORKDIR /workspace

# 4. Copy and install Python dependencies
COPY requirements.txt /workspace/
RUN pip3 --default-timeout=100 install --upgrade pip wheel 'setuptools<66.0.0' \
 && pip3 --default-timeout=100 install --only-binary=grpcio grpcio \
 && pip3 --default-timeout=100 install -r requirements.txt

# 5. Copy the rest of the project
COPY . /workspace

# 6. Expose TensorBoard port (optional)
EXPOSE 6006

# 7. Default to bash for interactive use
ENTRYPOINT ["/bin/bash"]
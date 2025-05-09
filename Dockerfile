# 1. Base image with CUDA and cuDNN
FROM nvidia/cuda:12.4.0-cudnn8-devel-ubuntu20.04

# 2. Install OS-level prerequisites
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-pip python3-dev git && \
    rm -rf /var/lib/apt/lists/*

# 3. Set up a working directory
WORKDIR /De_Novo_drug_discovery

# 4. Copy your code and pinned deps
COPY requirements.txt /De_Novo_drug_discovery
COPY De_Novo_drug_discovery.ipynb /De_Novo_drug_discovery/
# (Also copy any helper scripts or data folders you use)

# 5. Install Python packages
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

# 6. Expose TensorBoard port (optional)
EXPOSE 6006

# 7. Default to bash so you can launch things interactively
ENTRYPOINT ["/bin/bash"]
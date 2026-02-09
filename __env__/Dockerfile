# Use a base image
#CUDA 12.8
FROM nvcr.io/nvidia/pytorch:25.03-py3 

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /build
COPY . .

# Install dependencies
RUN apt update && apt install -y software-properties-common \
    build-essential \
    ffmpeg libsm6 libxext6 \
    cmake \
    git \
    libjpeg-dev \
    libpng-dev \
    zip \
    unzip \
    tmux \
    curl \
    wget \
    zsh \
    fonts-powerline
    
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


RUN cd build/zsh && bash run.sh

RUN pip install --upgrade pip

# Install requirements list
RUN pip install -r requirements.txt
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
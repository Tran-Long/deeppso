# Use a base image
FROM nvcr.io/nvidia/pytorch:25.08-py3

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
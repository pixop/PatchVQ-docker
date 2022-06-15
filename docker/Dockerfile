FROM nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04

RUN apt update \
    && apt install -y ffmpeg wget git python3.6 python3-pip unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U pip setuptools wheel \
    && pip3 install fastai==2.0.18 cached-property loguru pillow==8.2 seaborn

RUN git clone https://github.com/pixop/PatchVQ-docker

WORKDIR PatchVQ-docker

RUN wget https://github.com/baidut/PatchVQ/releases/download/v0.1/RoIPoolModel-fit.10.bs.120.pth
RUN wget https://github.com/baidut/PatchVQ/releases/download/v0.1/fastai-r3d18_K_200ep.pth
RUN wget https://github.com/baidut/PatchVQ/releases/download/v0.1/release.zip \
    && unzip release.zip \
    && rm release.zip

RUN python3 -c 'import torchvision; torchvision.models.resnet18(pretrained=True)'

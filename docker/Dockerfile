FROM nvcr.io/nvidia/pytorch:20.03-py3

RUN git clone https://github.com/pixop/PatchVQ-docker

WORKDIR PatchVQ-docker

RUN pip install fastai==2.0.18 cached-property loguru pillow==8.2 Cython seaborn \
    && apt update \
    && apt install -y ffmpeg wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/baidut/PatchVQ/releases/download/v0.1/RoIPoolModel-fit.10.bs.120.pth
RUN wget https://github.com/baidut/PatchVQ/releases/download/v0.1/fastai-r3d18_K_200ep.pth

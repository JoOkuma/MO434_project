FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        libglib2.0-0 \
        software-properties-common \
        python3-dev \
        python3-pip \
        python3-tk \
        libsm6 libxrender1 libfontconfig1

ADD requirements.txt /tmp/.

RUN pip3 install --upgrade pip
RUN pip3 install setuptools wheel
RUN pip3 install torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install -r /tmp/requirements.txt

WORKDIR /tmp

CMD [ "/bin/bash" ]


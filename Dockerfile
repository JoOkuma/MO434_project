FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        libglib2.0-0 \
        software-properties-common \
        python3.7-dev \
        python3-pip \
        python3-tk \
        libsm6 libxrender1 libfontconfig1

RUN pip3 install --upgrade pip                                                                                                                                                                             
RUN pip3 install setuptools
RUN pip3 install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install pytorch-lightning
RUN pip3 install adabound
RUN pip3 install opencv-python==3.4.8.29

WORKDIR /tmp

CMD [ "/bin/bash" ]


FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04 AS app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update -y && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    git-lfs \
    wget \
    vim \
    libsndfile1 \
    ccache \
    software-properties-common \
    poppler-utils \
    build-essential \
    libgl1 \
    libreoffice \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip install --no-cache-dir --upgrade pip

WORKDIR /env
COPY ./ckpts /env/ckpts/

RUN python3 -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
COPY ./requirements.txt /env/
RUN pip install -r requirements.txt

COPY ./app /env/app/
COPY ./entrypoint.sh /env

RUN chmod +x ./entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]
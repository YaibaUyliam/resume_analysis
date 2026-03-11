FROM python:3.12.3-slim AS app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get upgrade -y && apt-get install -y --no-install-recommends \
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

WORKDIR /env
COPY ./ckpts /env/ckpts/

RUN pip install --no-cache-dir --upgrade pip
# Read config from pyproject.toml
RUN python3 -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
COPY ./requirements.txt /env/
RUN pip install -r requirements.txt

COPY ./app /env/app/
COPY ./entrypoint.sh /env

RUN chmod +x ./entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]
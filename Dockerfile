# Stage 1: Build app env
FROM yaibawiliam/ollama:v0.12.6.dev AS app

RUN <<EOF
apt update -y && apt upgrade -y && apt install -y --no-install-recommends  \
    git \
    git-lfs \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    vim \
    libsndfile1 \
    ccache \
    software-properties-common \
    poppler-utils \
    build-essential \
&& rm -rf /var/lib/apt/lists/*
EOF
RUN ln -s /usr/bin/python3 /usr/bin/python

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /env

# Stage 2: Pull models
FROM yaibawiliam/ollama:v0.12.6.dev AS models
RUN nohup bash -c "ollama serve &" && sleep 5 && ollama pull qwen2.5:14b-instruct-q5_K_M
RUN ollama pull qwen3-embedding:0.6b-fp16
# Final stage
FROM app
COPY --from=models /root/.ollama/models /root/.ollama/models
COPY ./pyproject.toml /env
RUN uv sync
COPY . /env
RUN chmod +x ./entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]

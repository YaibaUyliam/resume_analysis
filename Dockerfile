# Stage 1: Build app env
FROM ollama/ollama:v0.1.test AS app

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
COPY ./pyproject.toml /env
RUN uv sync

# Stage 2: Pull models
FROM ollama/ollama:v0.1.test AS models
RUN nohup bash -c "ollama serve &" && sleep 5 && ollama pull qwen2.5:14b-instruct-q5_K_M

# Final stage
FROM app
COPY --from=models /root/.ollama/models /root/.ollama/models
COPY . /env
RUN chmod +x ./entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]

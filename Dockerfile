# Stage 1
FROM ollama/ollama:0.11.6 AS models
RUN nohup bash -c "ollama serve &" && sleep 5 && ollama pull deepseek-r1:14b

#Stage 2
FROM ollama/ollama:0.11.6

# Copy model từ stage 1 sang
COPY --from=models /root/.ollama/models /root/.ollama/models

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
# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

# Copy code app
WORKDIR /env
COPY ./pyproject.toml /env
RUN uv sync
COPY . /env

RUN chmod +x ./entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]

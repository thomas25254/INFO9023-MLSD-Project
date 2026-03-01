# ----------------------------
# Base Image
# ----------------------------
FROM python:3.11-slim

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# ----------------------------
# System Dependencies
# ----------------------------
RUN apt-get update && apt-get install -y \
    ffmpeg \
    wget \
    unzip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------
# Install uv
# ----------------------------
RUN curl -Ls https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# ----------------------------
# Set Workdir
# ----------------------------
WORKDIR /app

# ----------------------------
# Copy dependency files first (better layer caching)
# ----------------------------
COPY pyproject.toml uv.lock ./

# Install Python deps
RUN uv sync --frozen

# ----------------------------
# Download Vosk Models
# ----------------------------
RUN mkdir -p /opt/models && \
    cd /opt/models && \
    wget https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip && \
    unzip vosk-model-en-us-0.22.zip && \
    rm vosk-model-en-us-0.22.zip && \
    wget https://alphacephei.com/vosk/models/vosk-model-spk-0.4.zip && \
    unzip vosk-model-spk-0.4.zip && \
    rm vosk-model-spk-0.4.zip

# ----------------------------
# Download Dataset
# ----------------------------
RUN mkdir -p /opt/dataset && \
    cd /opt/dataset && \
    wget https://openslr.trmal.net/resources/12/dev-clean.tar.gz && \
    tar -xzvf dev-clean.tar.gz && \
    rm dev-clean.tar.gz

# ----------------------------
# Copy rest of project
# ----------------------------
COPY . .

# ----------------------------
# Expose Jupyter port
# ----------------------------
EXPOSE 8888

# ----------------------------
# Run Jupyter
# ----------------------------
CMD ["uv", "run", "jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]

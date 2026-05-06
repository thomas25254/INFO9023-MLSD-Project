FROM mirror.gcr.io/library/python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements-pipeline.txt .
RUN pip install --no-cache-dir -r requirements-pipeline.txt

ENTRYPOINT ["bash"]

FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY motion.py record.py ./
COPY samples/sample.mp4 samples/sample.mp4

RUN useradd --create-home appuser \
    && mkdir -p /app/captures \
    && chown -R appuser:appuser /app
USER appuser

ENV ENV="prod" \
    VIDEO_SOURCE="samples/sample.mp4" \
    WARMUP_TIME="0"

CMD ["python", "record.py"]

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential curl && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/gotutiyan/gector.git .

RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir "transformers==4.49.0"

CMD ["bash"]
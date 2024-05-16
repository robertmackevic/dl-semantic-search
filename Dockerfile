FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY ./src ./src
COPY ./models ./models
COPY main.py .
COPY requirements.txt .

ARG MONGO_CONNECTION_STRING
ENV MONGO_CONNECTION_STRING=$MONGO_CONNECTION_STRING

RUN apt-get update && \
    pip install -r requirements.txt && \
    # Cleanup
    apt-get clean && rm -rf /var/lib/apt/lists/*

CMD ["python", "main.py"]
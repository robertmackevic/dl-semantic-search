FROM python:3.12

WORKDIR /app

COPY ./src ./src
COPY ./models ./models
COPY main.py .
COPY requirements.txt .

RUN apt-get update && \
    pip install -r requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

CMD ["python", "main.py"]
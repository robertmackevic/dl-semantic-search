# About

This is an example application of a semantic search tool that utilizes an NLP-based transformer
neural network for text feature extraction to perform efficient vector searches based on queries using RAG.

This implementation utilizes data taken from
[arXiv STEM scholarly articles](https://data.world/liz-friedman/arxiv-stem-scholarly-articles).
This is a large corpus of metadata gathered from 1.7 million+ arXiv articles (2.61 GB). For this application,
the metadata of 5000 latest articles (up to 2020-8-14) along with generated text embeddings were stored in MongoDB.

To create text embeddings, General Text Embedding (GTE) models were utilized from
[Huggingface](https://huggingface.co/thenlper).
Database vector search works on embeddings created by the [GTE-small](https://huggingface.co/thenlper/gte-small)
model variant, which has an embedding dimension of 384. Any other scale model can be used, but for that,
text embeddings need to be recomputed, database refilled, and vector search index configuration updated.

The application also features RAG-to-LLM pipeline,
which utilizes the OpenAI API to send calls to the GPT-3.5-turbo model.
This feature is optional and only available if you have a valid `OPENAI_API_KEY` present in your environment.

# Docker

This application is made available to run through Docker services.

Docker repository: https://hub.docker.com/r/robertmackevic/semantic-search

The application utilized a Pytorch CUDA base image, meaning it supports GPU inference.
However, the GTE-small model used is very lightweight,
so it runs fine on the CPU as well, and the GPU is purely optional.

## Running

Pull the latest image from the repository:

```
docker pull robertmackevic/semantic-search:latest
```

Run the application: [CPU]

```
docker run --name semantic-search --rm -ti robertmackevic/semantic-search:latest
```

Run the application: [GPU]

```
docker run --gpus=all --name semantic-search --rm -ti robertmackevic/semantic-search:latest
```

Run the application: [CPU and GPT support] (This requires `OPENAI_API_KEY` to be present in the host machine env.)

```
docker run --name semantic-search --env OPENAI_API_KEY=%OPENAI_API_KEY% --rm -ti robertmackevic/semantic-search:latest
```

* Windows: `%OPENAI_API_KEY%`
* Linux: `$OPENAI_API_KEY`

### Arguments

* `--gpus=all`: Instructs Docker to make all available GPUs on the host machine accessible within the container.
* `--name`: Assigns the given name (semantic-search in this case) to the Docker container.
* `--env`: Sets an environment variable inside the Docker container. In this case we use it to set `OPENAI_API_KEY`.
* `--rm`: Instructs Docker to automatically remove the container upon exit.
* `-it`: Allows interactive communication with the container, enabling features like keyboard input.

## Building

For building (and pushing) the image, a helpful Python script was created: `build_docker_image.py`.
This script streamlines the process of creating the Docker image by automating some aspects, such as
downloading the required models and forwarding environment variables.
However, the image can also be built and pushed manually with the following steps:

1. The GTE model needs to be downloaded and saved in a specific format.
   This can be done by running the following piece of code:

```python
from sentence_transformers import SentenceTransformer
from src.paths import MODEL_DIR

SentenceTransformer(model_name_or_path="thenlper/gte-small", cache_folder=MODEL_DIR)
```

2. Run the build command in the repository root directory:

```
docker build -t semantic-search --build-arg MONGO_CONNECTION_STRING=%MONGO_CONNECTION_STRING% .
```

For the application to work we need to pass in a valid `MONGO_CONNECTION_STRING`.
We do this by taking the value from the host machine's environment.
However, unlike `OPENAI_API_KEY`, this variable is crucial for the application to function and must remain the same.
That's why we're adding it during build time.

* Windows: `%MONGO_CONNECTION_STRING%`
* Linux: `$MONGO_CONNECTION_STRING`

3. Tag the image after it's built:

```
docker tag semantic-search:latest robertmackevic/semantic-search:latest
```

4. Push the image to the repository:

```
docker push robertmackevic/semantic-search:latest
```
"""
Script for automated Docker image building (and pushing).

Requirements:
* MONGO_CONNECTION_STRING
* [Optional] Push permission to `robertmackevic` Docker Hub repository.
"""
import subprocess
from argparse import ArgumentParser, Namespace
from os import environ

from sentence_transformers import SentenceTransformer

from src.embedding import GTE
from src.paths import MODEL_DIR

DOCKER_IMAGE_NAME = "semantic-search"
DOCKER_IMAGE_REPOSITORY = f"robertmackevic/{DOCKER_IMAGE_NAME}:latest"


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--gte", type=GTE, required=False, default=GTE.SMALL)
    parser.add_argument("--push", required=False, action="store_true", default=False)
    return parser.parse_args()


def run(gte: GTE, push: bool) -> None:
    SentenceTransformer(model_name_or_path=gte.value, cache_folder=MODEL_DIR)
    subprocess.run([
        "docker", "build",
        "-t", DOCKER_IMAGE_NAME,
        "--build-arg", f"MONGO_CONNECTION_STRING={environ.get("MONGO_CONNECTION_STRING")}",
        ".",
    ])

    if push:
        subprocess.run(["docker", "tag", f"{DOCKER_IMAGE_NAME}:latest", DOCKER_IMAGE_REPOSITORY])
        subprocess.run(["docker", "push", DOCKER_IMAGE_REPOSITORY])


if __name__ == "__main__":
    run(**vars(parse_args()))

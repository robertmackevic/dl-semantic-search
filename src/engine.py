from os import environ
from typing import List, Mapping, Any, Optional

from pandas import DataFrame
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from src.embedding import EmbeddingModel
from src.gpt import GPTClient


class SearchEngine:
    GTE_LARGE = "thenlper/gte-large"
    GTE_BASE = "thenlper/gte-base"
    GTE_SMALL = "thenlper/gte-small"
    GPT3_5 = "gpt-3.5-turbo"
    DEFAULT_NUM_RESULTS = 3
    DEFAULT_NUM_CANDIDATES = 150

    def __init__(
            self,
            openai_key: Optional[str] = None,
            embedding_model_name: str = GTE_BASE,
            gpt_version: str = GPT3_5,
            num_candidates: int = DEFAULT_NUM_CANDIDATES,
            max_results: int = DEFAULT_NUM_RESULTS,
    ) -> None:
        self.client = MongoClient(environ.get("MONGO_CONNECTION_STRING"), server_api=ServerApi("1"))

        try:
            self.client.admin.command("ping")
            print("Successfully connected to MongoDB!")

        except Exception as exception:
            print(exception)
            exit(1)

        self.collection = self.client["embeddings"]["articles"]

        self.num_candidates = num_candidates
        self.max_results = max_results
        self.embedding_model = EmbeddingModel(embedding_model_name)
        self.gpt_client = GPTClient(openai_key, gpt_version)

    def fill_embedding_database(self, dataframe: DataFrame) -> None:
        self.collection.delete_many({})
        self.collection.insert_many(dataframe.to_dict("records"))

    def _vector_search(self, query: str) -> List[Mapping[str, Any]]:
        query_embedding = self.embedding_model.get_embedding(query)

        results = self.collection.aggregate([
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "queryVector": query_embedding,
                    "path": "embedding",
                    "numCandidates": self.num_candidates,
                    "limit": self.max_results,
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "authors": 1,
                    "title": 1,
                    "journal-ref": 1,
                    "abstract": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ])
        return list(results)

    def semantic_search(self, query: str, use_gpt: bool) -> str:
        results = self._vector_search(query)
        response = ""

        for result in results:
            response += (
                f"Title: {result.get("title", "N/A")}\n"
                f"Authors: {result.get("authors", "N/A")}\n"
                f"Abstract: {result.get("abstract", "N/A")}\n"
                f"Journal reference: {result.get("journal-ref", "N/A")}\n\n"
            )

        return self.gpt_client.prompt(query, response) if use_gpt else response

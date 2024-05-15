import re
import string
import warnings
from enum import Enum
from typing import List

from sentence_transformers import SentenceTransformer
from torch import cuda

from src.paths import MODEL_DIR

warnings.filterwarnings("ignore", category=FutureWarning)


class GTE(Enum):
    SMALL = "thenlper/gte-small"
    BASE = "thenlper/gte-base"
    LARGE = "thenlper/gte-large"


class EmbeddingModel:
    def __init__(self, model_version: GTE) -> None:
        self.device = "cuda" if cuda.is_available() else "cpu"
        self.model = SentenceTransformer(
            model_name_or_path=model_version.value,
            device=self.device,
            cache_folder=MODEL_DIR,
        )

    def get_embedding(self, text: str) -> List[float]:
        text = self._preprocess_text(text)

        if len(text) == 0:
            raise ValueError("Input text is empty.")

        embedding = self.model.encode(text)
        return embedding.tolist()

    @staticmethod
    def _preprocess_text(text: str) -> str:
        pipe = text.lower()
        pipe = re.sub(f"[{re.escape(string.punctuation)}\n\t]", "", pipe)
        pipe = " ".join(pipe.split()).strip()
        return pipe

import re
import string
import warnings
from typing import List

from sentence_transformers import SentenceTransformer
from torch import cuda

warnings.filterwarnings("ignore", category=FutureWarning)


class EmbeddingModel:
    def __init__(self, model_name: str) -> None:
        self.device = "cuda" if cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

    def get_embedding(self, text: str) -> List[float] | ValueError:
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

from openai import OpenAI
from retrying import retry


class GPTClient:
    GPT3_5 = "gpt-3.5-turbo"

    def __init__(self, model_version: str = GPT3_5) -> None:
        self.model_version = model_version
        self.client = OpenAI()

    @retry(stop_max_attempt_number=5, wait_fixed=1000)
    def prompt(self, query: str, text: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_version,
            messages=[
                {
                    "role": "system",
                    "content":
                        "Your role is to answer the query based on given information about research papers.\n"
                        "Don't add additional information and only use information from the provided papers.\n"
                        "When answering the query, give a short summary.\n"
                },
                {
                    "role": "user",
                    "content":
                        f"Query: {query}\n"
                        f"Proceed to answer the query based on given information about research papers:\n{text}"
                },
            ]
        )
        return response.choices[0].message.content

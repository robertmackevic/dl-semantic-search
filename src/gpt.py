from os import environ

from openai import OpenAI


class GPTClient:

    def __init__(self, version: str) -> None:
        self.version = version
        self.client = OpenAI(api_key=environ.get("OPENAI_API_KEY"))

    def prompt(self, query: str, text: str) -> str:
        response = self.client.chat.completions.create(
            model=self.version,
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

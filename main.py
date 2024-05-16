"""
Entry point of the semantic search tool application.
This is only an example program made for learning purposes.
The database used by the tool will be deleted in the future.

To use the RAG-to-GPT pipeline mode, enter a single TAB character as a query.
The displayed text should switch from `>>>` to `[gpt] >>>`

Requirements:
* MONGO_CONNECTION_STRING
* [Optional] OPENAI_API_KEY
"""
from argparse import Namespace, ArgumentParser
from os import environ

from src.engine import SearchEngine
from src.gpt import GPTClient


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--gpt-version", type=str, required=False, default=GPTClient.GPT3_5)
    parser.add_argument("--num-candidates", type=int, required=False, default=SearchEngine.DEFAULT_NUM_CANDIDATES)
    parser.add_argument("--num-results", type=int, required=False, default=SearchEngine.DEFAULT_NUM_RESULTS)
    return parser.parse_args()


def run(gpt_version: str, num_candidates: int, num_results: int) -> None:
    openai_key = environ.get("OPENAI_API_KEY")

    if openai_key is None:
        print(
            "[WARNING] OPENAI_API_KEY was not found in environment variables. "
            "The program will not be able to summarize semantic search results using GPT."
        )

    use_gpt = False
    search_engine = SearchEngine(
        gpt_client=None if openai_key is None else GPTClient(gpt_version),
        num_candidates=num_candidates,
        max_results=num_results
    )
    print("Semantic search engine initialized!")
    print("Embedding model device:", search_engine.embedding_model.device)

    try:
        while True:
            query = input(("[gpt] " if use_gpt else "") + ">>> ")

            if query == "\t":
                use_gpt = not use_gpt
                continue

            try:
                if use_gpt and openai_key is None:
                    print(
                        "Unable to use GPT summarization because OPENAI_API_KEY is not an environment variable. "
                        "Type and enter the TAB key to disable RAG-to-GPT pipeline."
                    )
                    continue

                print(search_engine.semantic_search(query, use_gpt=use_gpt))

            except ValueError:
                pass

    except KeyboardInterrupt:
        print("\nProgram terminated.")
        return


if __name__ == "__main__":
    run(**vars(parse_args()))

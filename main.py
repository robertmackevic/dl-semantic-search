from argparse import Namespace, ArgumentParser

from src.engine import SearchEngine


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--emb", type=str, required=False, default=SearchEngine.GTE_BASE)
    parser.add_argument("--gpt", type=str, required=False, default=SearchEngine.GPT3_5)
    parser.add_argument("--num-candidates", type=int, required=False, default=SearchEngine.DEFAULT_NUM_CANDIDATES)
    parser.add_argument("--num-results", type=int, required=False, default=SearchEngine.DEFAULT_NUM_RESULTS)
    return parser.parse_args()


def run(emb: str, gpt: str, num_candidates: int, num_results: int) -> None:
    search_engine = SearchEngine(emb, gpt, num_candidates, num_results)
    use_gpt = False

    try:
        while True:
            query = input(("[gpt] " if use_gpt else "") + ">>> ")

            if query == "\t":
                use_gpt = not use_gpt
                continue

            try:
                print(search_engine.semantic_search(query, use_gpt=use_gpt))

            except ValueError as exception:
                print(exception)

    except KeyboardInterrupt:
        print("Program terminated.")
        return


if __name__ == "__main__":
    run(**vars(parse_args()))

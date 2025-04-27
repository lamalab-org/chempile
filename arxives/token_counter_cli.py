import tiktoken
import fire
import pandas as pd


def count_tokens(file_path, model="gpt2"):
    # read_jsonl
    text_as_lists = pd.read_json(file_path, lines=True)
    # merge all in "text" column
    text = text_as_lists["text"].str.cat(sep="\n")
    encoding = tiktoken.get_encoding(model)
    tokens = encoding.encode(text)
    return len(tokens)


if __name__ == "__main__":
    fire.Fire(count_tokens)

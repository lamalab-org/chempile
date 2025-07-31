import os
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv("../.env", override=True)


def main(hf_token):
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is not set.")

    ds = load_dataset("jablonkagroup/ChemEmb")
    # First split: train/val/test (80/10/10)
    split_1 = ds["train"].train_test_split(test_size=0.1, seed=42)
    train = split_1["train"]
    test = split_1["test"]
    split_2 = test.train_test_split(test_size=0.5, seed=42)  # ~10% of total for val
    val = split_2["train"]
    test = split_2["test"]
    # Upload splits to HF
    from datasets import DatasetDict

    dataset_dict = DatasetDict({"train": train, "val": val, "test": test})
    dataset_dict.push_to_hub("jablonkagroup/ChemEmb", token=hf_token)


if __name__ == "__main__":
    HF_TOKEN = os.environ.get("HF_TOKEN")
    main(HF_TOKEN)

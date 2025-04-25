from datasets import load_dataset


def preprocess():
    dataset = load_dataset("kjappelbaum/chemnlp-mol-svg")
    # rename smiles to SMILES
    dataset = dataset.rename_column("smiles", "SMILES")
    df = dataset["train"].to_pandas()
    df.dropna(inplace=True)
    print(len(df))
    df.to_csv("data_clean.csv", index=False)


if __name__ == "__main__":
    preprocess()

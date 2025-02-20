import pandas as pd
from huggingface_hub import hf_hub_download

columns_to_keep = ["phase1", "T", "BigSMILES", "Mn", "f1", "Mw", "D"]


def process():
    df = hf_hub_download(repo_id="AdrianM0/block_polymers_morphology", filename="diblock.csv", repo_type="dataset")
    df = pd.read_csv(df)
    df.to_csv("data_clean.csv", index=False)

if __name__ == "__main__":
    process()

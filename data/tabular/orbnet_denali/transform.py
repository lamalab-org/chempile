from datasets import load_dataset
from rdkit import Chem
from joblib import Parallel, delayed

def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)

def process():
    dataset = load_dataset("kjappelbaum/chemnlp-orbnet-denali")
    df = dataset["train"].to_pandas()
    
    # Parallelize the canonicalization
    df["smiles"] = Parallel(n_jobs=4)(
        delayed(canonicalize_smiles)(s) for s in df["smiles"]
    )

    df = df.dropna()
    print(len(df))
    df.rename(columns={"smiles": "SMILES"}, inplace=True)
    df.to_csv("data_clean.csv", index=False)

if __name__ == "__main__":
    process()

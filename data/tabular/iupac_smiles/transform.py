import os
import fire
import pandas as pd
from datasets import load_dataset
from rdkit import Chem
from multiprocessing import Pool


def canonicalize_smiles(smiles):
    """Convert SMILES to canonical form"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, isomericSmiles=True)
        return None
    except:
        return None


def process_chunk(chunk):
    """Process a chunk of SMILES strings"""
    chunk["SMILES"] = chunk["SMILES"].apply(canonicalize_smiles)
    return chunk.dropna(subset=["SMILES"])


def process(debug=False, n_jobs=None):
    if n_jobs is None:
        n_jobs = 12

    print(f"Using {n_jobs} processes for parallelization")

    if not os.path.exists("combined_json.jsonl"):
        print("Loading dataset from HuggingFace...")
        dataset = load_dataset("kjappelbaum/chemnlp_iupac_smiles")
        df = pd.DataFrame(dataset["train"])
    else:
        print("Loading dataset from local file...")
        file = "combined_json.jsonl"
        df = pd.read_json(file, lines=True)

    print(f"Original dataset size: {len(df)}")

    if debug:
        print("Debug mode: sampling 1000 entries")
        df = df.sample(1000)

    # Split dataframe into chunks for parallel processing
    chunk_size = max(1, len(df) // n_jobs)
    chunks = [df.iloc[i : i + chunk_size] for i in range(0, len(df), chunk_size)]
    print(f"Processing {len(chunks)} chunks of data...")

    # Process chunks in parallel
    with Pool(processes=n_jobs) as pool:
        processed_chunks = pool.map(process_chunk, chunks)

    # Combine processed chunks
    processed_df = pd.concat(processed_chunks)
    print(f"After canonicalization and filtering: {len(processed_df)} entries")

    # Remove duplicates (based on canonical SMILES)
    processed_df.drop_duplicates(subset=["SMILES"], inplace=True)
    print(f"After removing duplicates: {len(processed_df)} entries")

    # Save to CSV
    processed_df.to_csv("data_clean.csv", index=False)
    print("Data saved to data_clean.csv")


if __name__ == "__main__":
    fire.Fire(process)

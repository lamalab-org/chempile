import os
from datasets import load_dataset, Dataset, concatenate_datasets
from dotenv import load_dotenv
from huggingface_hub import HfFolder


def sample_chemistry_data():
    """
    Samples data from chemistry-related datasets on Hugging Face.

    This function loads data from 'jablonkagroup/chempile-paper' and
    'jablonkagroup/chempile-education', standardizes their columns,
    samples 25,000 rows in total, and returns the result.

    Returns:
        Dataset: A Hugging Face Dataset object containing the sampled chemistry data.
    """
    # Define the datasets and their specific configuration names to load
    datasets_to_load = [
        ("jablonkagroup/chempile-paper", "arxiv-cond-mat.mtrl-sci_processed-default"),
        ("jablonkagroup/chempile-paper", "arxiv-physics.chem-ph_processed-default"),
        ("jablonkagroup/chempile-paper", "biorxiv_processed-default"),
        ("jablonkagroup/chempile-paper", "chemrxiv_processed-default"),
        ("jablonkagroup/chempile-paper", "euro_pmc_chemistry_abstracts-default"),
        ("jablonkagroup/chempile-paper", "euro_pmc_chemistry_papers-default"),
        ("jablonkagroup/chempile-paper", "medrxiv_processed-default"),
        ("jablonkagroup/chempile-education", "LibreText_Chemistry-default"),
    ]

    processed_datasets = []

    # Process each dataset individually to create a uniform structure
    for dset_name, config_name in datasets_to_load:
        # Load the dataset
        ds = load_dataset(dset_name, name=config_name, split="train")

        # 1. Keep only the "text" column
        # First, find all columns that are NOT "text"
        columns_to_remove = [col for col in ds.column_names if col != "text"]
        ds = ds.remove_columns(columns_to_remove)

        # 2. Create and add the new "metadata" column
        metadata_value = f"{dset_name.split('/')[1]}_{config_name}"
        ds = ds.add_column("metadata", [metadata_value] * len(ds))
        ds = ds.add_column("chemistry_content", [True] * len(ds))

        processed_datasets.append(ds)

    # Now, concatenation will work because all datasets have the same columns
    combined_chem_dataset = concatenate_datasets(processed_datasets)

    # Sample 25,000 rows from the combined dataset
    sampled_chem_dataset = combined_chem_dataset.shuffle(seed=42).select(range(25000))

    return sampled_chem_dataset


def sample_gutenberg_data():
    """
    Samples data from the Gutenberg English dataset on Hugging Face.

    This function loads data from 'sedthh/gutenberg_english', samples 25,000 rows,
    and formats it into a standardized structure.

    Returns:
        Dataset: A Hugging Face Dataset object containing the sampled Gutenberg data.
    """
    # Load the dataset
    gutenberg_dataset = load_dataset("sedthh/gutenberg_english", split="train")

    # Sample 25,000 rows
    sampled_gutenberg = gutenberg_dataset.shuffle(seed=42).select(range(25000))

    # Prepare the data
    gutenberg_data = {
        "text": sampled_gutenberg["TEXT"],
        "chemistry_content": [False] * len(sampled_gutenberg),
        "metadata": [
            "Project Gutenberg english " + str(meta)
            for meta in sampled_gutenberg["METADATA"]
        ],
    }

    return Dataset.from_dict(gutenberg_data)


def main():
    """
    Main function to create, combine, and upload the dataset.

    This function orchestrates the sampling of chemistry and Gutenberg data,
    combines them, shuffles the resulting dataset, and uploads it to the
    Hugging Face Hub.
    """
    # Load environment variables
    load_dotenv("../.env")
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise ValueError(
            "Hugging Face token not found. Please add it to the ../.env file."
        )

    # Authenticate with Hugging Face
    HfFolder.save_token(hf_token)

    # Sample the data
    print("Sampling chemistry data...")
    chem_dataset = sample_chemistry_data()
    print("Sampling Gutenberg data...")
    gutenberg_dataset = sample_gutenberg_data()

    # Combine and shuffle the datasets
    print("Combining and shuffling datasets...")
    final_dataset = concatenate_datasets([chem_dataset, gutenberg_dataset]).shuffle(
        seed=42
    )

    # Upload the dataset to the Hugging Face Hub
    print("Uploading dataset to Hugging Face Hub...")
    repo_name = "jablonkagroup/ChemEmb"
    final_dataset.push_to_hub(repo_name, private=True)

    print(f"Dataset successfully uploaded to {repo_name}")


if __name__ == "__main__":
    main()

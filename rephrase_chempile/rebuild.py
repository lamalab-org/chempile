#!/usr/bin/env python3
"""
Script to download jablonkagroup/chempile-reasoning dataset,
add a new 'text' column, and upload back to HuggingFace.
"""

import os
from datasets import load_dataset, DatasetDict
from huggingface_hub import login
from dotenv import load_dotenv


def create_text_column(example):
    """
    Create the text column by combining title, question, and answer.
    """
    title = example.get("title", "")
    q = example.get("q", "")  # Assuming 'q' is the question column name
    a = example.get("a", "")  # Assuming 'a' is the answer column name

    text = f"Title: {title}\n\nQuestion: {q}\n\nAnswer: {a}"
    example["text"] = text
    return example


def main():
    # Load environment variables from ../.env
    load_dotenv("../.env")

    # Login to HuggingFace
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("Successfully logged in to HuggingFace")
    else:
        print("Error: HF_TOKEN not found in ../.env file")
        return

    # Download the original dataset
    print("Downloading original dataset...")
    try:
        dataset = load_dataset(
            "jablonkagroup/chempile-reasoning", "physics_stackexchange-raw_data"
        )
        print(f"Dataset loaded successfully: {dataset}")

        # Print column names to verify
        if len(dataset) > 0:
            first_split = list(dataset.keys())[0]
            print(f"Available columns: {dataset[first_split].column_names}")
            print(f"First example keys: {list(dataset[first_split][0].keys())}")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Process each split to add the text column
    print("Adding 'text' column to all splits...")
    processed_dataset = DatasetDict()

    for split_name, split_data in dataset.items():
        print(f"Processing {split_name} split ({len(split_data)} examples)...")

        # Add the text column
        processed_split = split_data.map(
            create_text_column, desc=f"Adding text column to {split_name}"
        )

        processed_dataset[split_name] = processed_split

        # Show a sample of the new text column
        if len(processed_split) > 0:
            print(f"Sample from {split_name} split:")
            print(f"Text preview: {processed_split[0]['text'][:200]}...")
            print()

    # Upload the processed dataset back to the original repository
    repo_name = "jablonkagroup/chempile-reasoning"
    config_name = "physics_stackexchange-raw_data"

    print(f"Uploading processed dataset back to {repo_name} (config: {config_name})...")
    try:
        processed_dataset.push_to_hub(repo_name, config_name=config_name)
        print(f"Dataset successfully uploaded back to {repo_name}")

    except Exception as e:
        print(f"Error uploading dataset: {e}")
        print("Make sure you have write permissions to the repository.")
        return

    # Verify the upload by loading a small sample
    print("Verifying upload...")
    try:
        verification_dataset = load_dataset(
            repo_name,
            config_name,
            split="train[:5]",  # Load just 5 examples for verification
        )
        print(
            f"Verification successful! Updated dataset has {len(verification_dataset.column_names)} columns:"
        )
        print(f"Columns: {verification_dataset.column_names}")

    except Exception as e:
        print(f"Warning: Could not verify upload: {e}")


if __name__ == "__main__":
    main()

# Example usage:
# python script.py
#
# Make sure you have a ../.env file with:
# HF_TOKEN=your_huggingface_token

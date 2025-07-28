#!/usr/bin/env python3
"""
Script to download a dataset from Hugging Face Hub and upload it to another dataset
with a specific configuration name while preserving original splits.
"""

from datasets import load_dataset, DatasetDict
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv("../.env", override=True)  # Load environment variables from .env file


def transfer_dataset(
    source_dataset: str,
    target_dataset: str,
    config_name: str,
    token: Optional[str] = None,
):
    """
    Download a dataset and upload it to another dataset with a specific config name.

    Args:
        source_dataset: Name of the source dataset (e.g., 'jablonkagroup/spectra_reasoning_deepseek_mcq')
        target_dataset: Name of the target dataset (e.g., 'jablonkagroup/chempile-reasoning')
        config_name: Configuration name for the target dataset
        token: Hugging Face token (optional, will use HF_TOKEN env var if not provided)
    """

    # Use token from environment if not provided
    if token is None:
        token = os.getenv("HF_TOKEN")
        if not token:
            print("Warning: No token provided and HF_TOKEN not found in environment.")
            print("You may need to authenticate with `huggingface-cli login` first.")

    print(f"Downloading dataset: {source_dataset}")

    try:
        # Load the entire dataset with all splits
        dataset = load_dataset(source_dataset, token=token)

        print(f"Successfully loaded dataset with splits: {list(dataset.keys())}")
        print("Dataset info:")
        for split_name, split_data in dataset.items():
            print(f"  - {split_name}: {len(split_data)} examples")

        # Check for splits and rename 'validation' to 'val' if needed
        splits = set(dataset.keys())
        required_splits = {"train", "test", "val"}
        if "validation" in splits and "val" not in splits:
            print("Renaming 'validation' split to 'val'...")
            # Create a new DatasetDict with 'val' instead of 'validation'
            new_splits = {}
            for split_name, split_data in dataset.items():
                if split_name == "validation":
                    new_splits["val"] = split_data
                else:
                    new_splits[split_name] = split_data
            dataset = DatasetDict(new_splits)
            print(f"Splits after renaming: {list(dataset.keys())}")

        # Warn if required splits are missing
        splits = set(dataset.keys())
        for req in required_splits:
            if req not in splits:
                print(f"Warning: Required split '{req}' not found in dataset.")

        # Upload to target dataset with specified config name
        print(f"Uploading to {target_dataset} with config '{config_name}'...")

        dataset.push_to_hub(
            target_dataset,
            config_name=config_name,
            token=token,
            private=False,  # Since you mentioned the dataset already exists publicly
        )

        print(
            f"Successfully uploaded dataset to {target_dataset} with config '{config_name}'"
        )
        print("All original splits have been preserved (with 'val' if needed).")

    except Exception as e:
        print(f"Error during transfer: {e}")
        raise


def main():
    """Main function to execute the dataset transfer."""

    # Configuration
    SOURCE_DATASET = "jablonkagroup/spectra_reasoning_deepseek_mcq"
    TARGET_DATASET = "jablonkagroup/chempile-reasoning"
    CONFIG_NAME = "spectra_reasoning_deepseek_mcq-default"

    print("Dataset Transfer Script")
    print("=" * 50)
    print(f"Source: {SOURCE_DATASET}")
    print(f"Target: {TARGET_DATASET}")
    print(f"Config: {CONFIG_NAME}")
    print("=" * 50)

    # Execute transfer
    transfer_dataset(
        source_dataset=SOURCE_DATASET,
        target_dataset=TARGET_DATASET,
        config_name=CONFIG_NAME,
    )


if __name__ == "__main__":
    main()

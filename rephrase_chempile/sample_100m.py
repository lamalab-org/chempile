from datasets import load_dataset, Dataset
import random
import json
from typing import List, Dict, Any
import tiktoken
import os


def count_tokens(text: str, encoding_name: str = "gpt2") -> int:
    """Count tokens in text using tiktoken."""
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception:
        raise ValueError(
            f"Could not count tokens for text: {text[:50]}... (encoding: {encoding_name})"
        )


def extract_text_from_messages(messages: List[Dict[str, Any]]) -> str:
    """Extract all text content from messages list."""
    text_parts = []
    for message in messages:
        if isinstance(message, dict) and "content" in message:
            text_parts.append(str(message["content"]))
    return " ".join(text_parts)


def load_and_sample_config(config_name: str, target_tokens: int) -> List[Dict]:
    """Load a config and sample rows to reach target token count."""
    print(f"Loading config: {config_name}")

    try:
        dataset = load_dataset(
            "jablonkagroup/chempile-instruction", config_name, split="train"
        )
        print(f"Loaded {len(dataset)} rows from {config_name}")
    except Exception as e:
        print(f"Error loading {config_name}: {e}")
        return []

    # Calculate tokens for each row
    rows_with_tokens = []
    total_available_tokens = 0

    for i, row in enumerate(dataset):
        if i % 1000 == 0:
            print(f"Processing row {i}/{len(dataset)} for {config_name}")

        messages = row.get("messages", [])
        if messages:
            text = extract_text_from_messages(messages)
            token_count = count_tokens(text)
            rows_with_tokens.append(
                {"row": row, "tokens": token_count, "text": messages}
            )
            total_available_tokens += token_count

    print(f"Config {config_name}: {total_available_tokens:,} total tokens available")

    if total_available_tokens <= target_tokens:
        print(f"Using all data from {config_name} ({total_available_tokens:,} tokens)")
        return [item["row"] for item in rows_with_tokens]

    # Sample randomly until we reach target tokens
    print(f"Sampling from {config_name} to reach {target_tokens:,} tokens")
    random.shuffle(rows_with_tokens)

    sampled_rows = []
    current_tokens = 0

    for item in rows_with_tokens:
        if current_tokens + item["tokens"] <= target_tokens:
            sampled_rows.append(item["row"])
            current_tokens += item["tokens"]
        elif current_tokens < target_tokens:
            # Add this row even if it goes slightly over target
            sampled_rows.append(item["row"])
            current_tokens += item["tokens"]
            break

    print(
        f"Sampled {len(sampled_rows)} rows with {current_tokens:,} tokens from {config_name}"
    )
    return sampled_rows


def push_to_huggingface(
    sampled_data: List[Dict],
    total_tokens: int,
    push_to_hub: bool = True,
    hub_token: str = None,
):
    """Create and push dataset to HuggingFace Hub."""
    print("\n=== CREATING HUGGINGFACE DATASET ===")

    # Create HuggingFace dataset
    dataset = Dataset.from_list(sampled_data)

    print(f"Created dataset with {len(dataset)} samples")
    print(f"Dataset features: {dataset.features}")

    # Add dataset metadata
    dataset_card = ""

    if push_to_hub:
        try:
            print("Pushing dataset to HuggingFace Hub...")

            # Push to hub with authentication
            dataset.push_to_hub(
                "jablonkagroup/chempile-instruction-100M",
                token=hub_token,
                private=False,
                commit_message=f"Add sampled ChemPile dataset with {total_tokens:,} tokens",
            )

            print(
                "✅ Successfully pushed dataset to jablonkagroup/chempile-instruction-100M"
            )

            # Save dataset card separately
            with open("README.md", "w") as f:
                f.write(dataset_card)
            print("📄 Saved dataset card to README.md")

        except Exception as e:
            print(f"❌ Error pushing to HuggingFace Hub: {e}")
            print("💡 Make sure you:")
            print(
                "   1. Have the correct permissions for the jablonkagroup organization"
            )
            print("   2. Are logged in with: huggingface-cli login")
            print("   3. Or provide a token with: --hub_token YOUR_TOKEN")

            # Save locally as fallback
            print("\n💾 Saving dataset locally as fallback...")
            dataset.save_to_disk("chempile-instruction-100M")
            print("Saved to: ./chempile-instruction-100M/")
    else:
        # Save locally only
        print("💾 Saving dataset locally...")
        dataset.save_to_disk("chempile-instruction-100M")
        with open("chempile-instruction-100M/README.md", "w") as f:
            f.write(dataset_card)
        print("Saved to: ./chempile-instruction-100M/")

    return dataset


def main(push_to_hub: bool = True, hub_token: str = None):
    # Target: 100M tokens total
    TARGET_TOTAL_TOKENS = 100000000

    # Get available configs - try different approaches
    configs_to_use = []
    try:
        from datasets import get_dataset_config_names

        available_configs = get_dataset_config_names(
            "jablonkagroup/chempile-instruction"
        )
        print(f"Available configs: {available_configs}")
        configs_to_use = (
            available_configs[:3] if len(available_configs) >= 3 else available_configs
        )
    except Exception as e:
        print(f"Could not get config names automatically: {e}")
        # Fallback to common config names
        configs_to_use = ["default"]
        print("Using fallback config: ['default']")

    if len(configs_to_use) == 0:
        print("No configs found. Exiting.")
        return

    tokens_per_config = TARGET_TOTAL_TOKENS // len(configs_to_use)
    print(f"Using configs: {configs_to_use}")
    print(f"Target tokens per config: {tokens_per_config:,}")

    # Sample from each config
    all_sampled_data = []
    total_tokens_collected = 0

    for config in configs_to_use:
        sampled_rows = load_and_sample_config(config, tokens_per_config)

        # Count actual tokens collected
        config_tokens = 0
        for row in sampled_rows:
            messages = row.get("messages", [])
            if messages:
                text = extract_text_from_messages(messages)
                config_tokens += count_tokens(text)

        all_sampled_data.extend(sampled_rows)
        total_tokens_collected += config_tokens
        print(f"Collected {config_tokens:,} tokens from {config}")

    print("\n=== SUMMARY ===")
    print(f"Total rows sampled: {len(all_sampled_data):,}")
    print(f"Total tokens collected: {total_tokens_collected:,}")
    print(f"Target tokens: {TARGET_TOTAL_TOKENS:,}")
    print(
        f"Percentage of target: {(total_tokens_collected/TARGET_TOTAL_TOKENS)*100:.2f}%"
    )

    # Save the sampled data locally as JSON backup
    output_file = "chempile_sampled_100M.json"
    print(f"\nSaving sampled data to {output_file}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_sampled_data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_sampled_data)} samples to {output_file}")

    # Create a summary file
    summary = {
        "total_rows": len(all_sampled_data),
        "total_tokens": total_tokens_collected,
        "target_tokens": TARGET_TOTAL_TOKENS,
        "configs_used": configs_to_use,
        "tokens_per_config_target": tokens_per_config,
    }

    with open("sampling_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Create and push HuggingFace dataset
    dataset = push_to_huggingface(
        all_sampled_data, total_tokens_collected, push_to_hub, hub_token
    )

    print("✅ Process complete!")
    return dataset


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(".env", override=True)
    HF_TOKEN = os.getenv("HF_TOKEN")

    # Set random seed for reproducibility
    random.seed(42)

    # Run main function
    push_to_hub = True
    main(push_to_hub=push_to_hub, hub_token=HF_TOKEN)

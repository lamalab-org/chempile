import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from datasets import get_dataset_config_names, load_dataset, Dataset, DatasetDict
import random
import tiktoken
import os
import json
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv("../.env", override=True)

client = AsyncOpenAI()

# Checkpoint configuration
CHECKPOINT_DIR = Path("./checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

SIMPLE_PROMPT = """Given the text below:

{document}

Generate exactly one question and its corresponding answer based solely on the content.
Note that the question should be directly answerable without requiring external knowledge or the original text.

Format your response as:
Question: [your question]
Answer: [your answer]

Provide only the formatted dialogue:"""

HARD_PROMPT = """From the text below, return a question-answer pair using these requirements:

1. Structure following the format: "Question: [your question] Answer: [your answer]"
2. The question should be answerable without requiring external knowledge or the original text.
3. Use esoteric vocabulary and complex syntax suitable for academics
4. Replace common terms with rare, technical alternatives

Text:

{document}

Provide only the reformatted pair:"""

WIKI_PROMPT = """From the text below, return a question-answer pair using these requirements:

1. Structure following the format: "Question: [your question] Answer: [your answer]"
2. The question should be answerable without requiring external knowledge or the original text.
3. Use formal, encyclopedic English resembling Wikipedia
4. Maintain factual accuracy and neutral tone

Text:

{document}

Output only the formatted pair:"""

BASE_QA_PROMPT = """From the text below extract a question-answer pair using:

- The rephrased text must contain one "Question:" and "Answer:" pair. Follow the format: "Question: [your question] Answer: [your answer]"
- The question should be answerable without requiring external knowledge or the original text.

Text:

{document}

Provide only the formatted question-answer pair:"""


def get_checkpoint_path(dataset_name: str, config: str, split_name: str) -> Path:
    """Generate checkpoint file path for a specific dataset/config/split."""
    safe_dataset_name = dataset_name.replace("/", "-")
    return CHECKPOINT_DIR / f"{safe_dataset_name}_{config}_{split_name}.pkl"


def get_progress_path(dataset_name: str) -> Path:
    """Generate progress tracking file path for a dataset."""
    safe_dataset_name = dataset_name.replace("/", "-")
    return CHECKPOINT_DIR / f"{safe_dataset_name}_progress.json"


def save_checkpoint(
    data: Dict[str, List], dataset_name: str, config: str, split_name: str
):
    """Save processed data to checkpoint file."""
    checkpoint_path = get_checkpoint_path(dataset_name, config, split_name)
    try:
        with open(checkpoint_path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")


def load_checkpoint(
    dataset_name: str, config: str, split_name: str
) -> Optional[Dict[str, List]]:
    """Load processed data from checkpoint file."""
    checkpoint_path = get_checkpoint_path(dataset_name, config, split_name)
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "rb") as f:
                data = pickle.load(f)
            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
    return None


def load_all_checkpoints_for_dataset(dataset_name: str) -> Dict[str, Dict[str, List]]:
    """Load all existing checkpoint files for a dataset and organize by split."""
    safe_dataset_name = dataset_name.replace("/", "-")
    pattern = f"{safe_dataset_name}_*_*.pkl"
    checkpoint_files = list(CHECKPOINT_DIR.glob(pattern))

    split_data = {}

    for checkpoint_file in checkpoint_files:
        try:
            # Parse filename to extract config and split
            filename = checkpoint_file.stem
            parts = filename.split("_")
            if len(parts) < 3:
                continue

            # Reconstruct config and split (accounting for potential underscores in names)
            split_name = parts[-1]

            # Load the checkpoint data
            with open(checkpoint_file, "rb") as f:
                data = pickle.load(f)

            logger.info(
                f"Loaded checkpoint: {checkpoint_file} ({len(data.get('text', []))} items)"
            )

            # Organize by split name
            if split_name not in split_data:
                split_data[split_name] = {
                    "original_document": [],
                    "text": [],
                    "question": [],
                    "answer": [],
                    "metadata": [],
                }

            # Append data to the split
            for key in ["original_document", "text", "question", "answer", "metadata"]:
                if key in data:
                    split_data[split_name][key].extend(data[key])

        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_file}: {e}")
            continue

    return split_data


def save_progress(dataset_name: str, progress_data: Dict[str, Any]):
    """Save progress information for a dataset."""
    progress_path = get_progress_path(dataset_name)
    try:
        with open(progress_path, "w") as f:
            json.dump(progress_data, f, indent=2)
        logger.info(f"Progress saved: {progress_path}")
    except Exception as e:
        logger.error(f"Failed to save progress: {e}")


def load_progress(dataset_name: str) -> Dict[str, Any]:
    """Load progress information for a dataset."""
    progress_path = get_progress_path(dataset_name)
    if progress_path.exists():
        try:
            with open(progress_path, "r") as f:
                progress_data = json.load(f)
            logger.info(f"Progress loaded: {progress_path}")
            return progress_data
        except Exception as e:
            logger.error(f"Failed to load progress: {e}")
    return {"completed_configs": [], "completed_splits": {}}


def save_final_dataset(dataset_dict: DatasetDict, dataset_name: str):
    """Save the final processed dataset locally."""
    safe_dataset_name = dataset_name.replace("/", "-")
    final_path = CHECKPOINT_DIR / f"{safe_dataset_name}_final"
    try:
        dataset_dict.save_to_disk(str(final_path))
        logger.info(f"Final dataset saved locally: {final_path}")
    except Exception as e:
        logger.error(f"Failed to save final dataset: {e}")


def load_final_dataset(dataset_name: str) -> Optional[DatasetDict]:
    """Load the final processed dataset from local storage."""
    safe_dataset_name = dataset_name.replace("/", "-")
    final_path = CHECKPOINT_DIR / f"{safe_dataset_name}_final"
    if final_path.exists():
        try:
            dataset_dict = DatasetDict.load_from_disk(str(final_path))
            logger.info(f"Final dataset loaded from: {final_path}")
            return dataset_dict
        except Exception as e:
            logger.error(f"Failed to load final dataset: {e}")
    return None


async def make_call(prompt: str, semaphore: asyncio.Semaphore) -> str:
    """Make async API call with rate limiting via semaphore."""
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in API call: {e}")
            return ""


async def check_configs(dataset_name: str) -> List[str]:
    """Check dataset configurations asynchronously."""
    loop = asyncio.get_event_loop()
    try:
        configs = await loop.run_in_executor(
            None, get_dataset_config_names, dataset_name
        )
        return configs
    except Exception as e:
        logger.error(f"Error retrieving configs for {dataset_name}: {e}")
        return []


def extract_question_answer(response: str) -> Optional[Dict[str, str]]:
    """
    Extracts the question and answer from the response string.
    Handles multi-line questions and answers.
    """
    if not response:
        return None

    response = response.strip()
    q_idx = response.find("Question:")
    a_idx = response.find("Answer:")

    if q_idx == -1 or a_idx == -1:
        logger.warning("Could not find 'Question:' or 'Answer:' in response.")
        return None

    question = response[q_idx + len("Question:") : a_idx].strip()
    answer = response[a_idx + len("Answer:") :].strip()

    if question and answer:
        return {"question": question, "answer": answer}
    else:
        logger.warning("Could not extract question and answer text.")
        return None


async def rephrase(
    document: str, semaphore: asyncio.Semaphore
) -> Optional[Dict[str, str]]:
    """Async version of rephrase function."""
    # Count tokens and truncate if necessary
    max_tokens = 120000
    num_tokens = count_tokens(document)

    if num_tokens > max_tokens:
        logger.info(
            f"Document has {num_tokens} tokens, truncating to {max_tokens} tokens."
        )
        encoding = tiktoken.encoding_for_model("gpt2")
        tokens = encoding.encode(document)
        truncated_tokens = tokens[:max_tokens]
        document = encoding.decode(truncated_tokens)
    else:
        logger.info(f"Document has {num_tokens} tokens, no truncation needed.")

    prompts = [SIMPLE_PROMPT, HARD_PROMPT, WIKI_PROMPT, BASE_QA_PROMPT]
    weights = [0.3, 0.1, 0.3, 0.3]  # Assigning less weight to HARD_PROMPT
    prompt = random.choices(prompts, weights=weights, k=1)[0]
    formatted_prompt = prompt.format(document=document)

    logger.info("Rephrasing document with selected prompt type")
    response = await make_call(formatted_prompt, semaphore)
    return extract_question_answer(response)


def count_tokens(text: str, model: str = "gpt2") -> int:
    """
    Counts the number of tokens in a given text using the specified model's tokenizer.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)
    return len(tokens)


async def process_split_documents(
    split_data,
    dataset_name: str,
    config: str,
    split_name: str,
    semaphore: asyncio.Semaphore,
    batch_size: int = 20,
) -> Dict[str, List]:
    """Process documents in a split with batching and checkpointing."""
    logger.info(f"Processing {len(split_data)} documents in split: {split_name}")

    # Try to load existing checkpoint
    checkpoint_data = load_checkpoint(dataset_name, config, split_name)
    if checkpoint_data:
        logger.info(
            f"Resuming from checkpoint for {split_name} with {len(checkpoint_data['original_document'])} existing items"
        )
        texts = checkpoint_data["original_document"]
        questions = checkpoint_data["question"]
        answers = checkpoint_data["answer"]
        metadata = checkpoint_data["metadata"]
        joined_texts = checkpoint_data["text"]
        start_idx = len(texts)
    else:
        texts = []
        questions = []
        answers = []
        metadata = []
        joined_texts = []
        start_idx = 0

    # Process remaining documents in batches
    for i in range(start_idx, len(split_data), batch_size):
        batch = split_data[i : i + batch_size]
        logger.info(
            f"Processing batch {i//batch_size + 1}/{(len(split_data) + batch_size - 1)//batch_size}"
        )

        # Create tasks for each document in the batch
        batch_tasks = []
        for j, document in enumerate(batch["text"]):
            actual_idx = i + j
            task = rephrase(document, semaphore)
            batch_tasks.append((task, actual_idx, document))

        # Execute batch tasks concurrently
        batch_results = await asyncio.gather(
            *[task[0] for task in batch_tasks], return_exceptions=True
        )

        # Process results
        for (_, actual_idx, document), result in zip(batch_tasks, batch_results):
            if isinstance(result, Exception):
                logger.error(f"Error processing document {actual_idx}: {result}")
                continue

            if result is None:
                logger.warning(
                    f"Failed to extract question/answer for document index {actual_idx} in split '{split_name}'"
                )
                continue

            questions.append(result.get("question", ""))
            answers.append(result.get("answer", ""))
            joined_texts.append(
                f"Question: {result.get('question', '')}\nAnswer: {result.get('answer', '')}"
            )
            texts.append(document)
            metadata.append(
                {
                    "original_dataset": dataset_name,
                    "original_config": config,
                    "original_split": split_name,
                    "original_index": actual_idx,
                }
            )

        # Save checkpoint after each batch
        checkpoint_data = {
            "original_document": texts,
            "text": joined_texts,
            "question": questions,
            "answer": answers,
            "metadata": metadata,
        }
        save_checkpoint(checkpoint_data, dataset_name, config, split_name)

    return {
        "original_document": texts,
        "text": joined_texts,
        "question": questions,
        "answer": answers,
        "metadata": metadata,
    }


async def process_single_dataset(
    dataset_name: str, max_concurrent_requests: int = 5
) -> DatasetDict:
    """
    Process a single dataset asynchronously with progress tracking and recovery.
    """
    logger.info(f"Processing dataset: {dataset_name}")

    # Check if final dataset already exists
    existing_dataset = load_final_dataset(dataset_name)
    if existing_dataset:
        logger.info(f"Found existing processed dataset for {dataset_name}")
        return existing_dataset

    # Load existing progress
    progress_data = load_progress(dataset_name)
    completed_configs = set(progress_data.get("completed_configs", []))
    completed_splits = progress_data.get("completed_splits", {})

    # Load all existing checkpoint data first
    logger.info(f"Loading existing checkpoint data for {dataset_name}")
    existing_split_data = load_all_checkpoints_for_dataset(dataset_name)

    # Create semaphore to limit concurrent API requests
    semaphore = asyncio.Semaphore(max_concurrent_requests)

    # Get all configs for this dataset
    configs = await check_configs(dataset_name)
    if not configs:
        raise ValueError(f"No configurations found for dataset: {dataset_name}")

    # Start with existing data
    processed_splits = existing_split_data.copy()

    for config in configs:
        if config in completed_configs:
            logger.info(f"Skipping already completed config: {config}")
            continue

        logger.info(f"Processing config: {config}")

        # Load the dataset with the specific config (run in executor to avoid blocking)
        loop = asyncio.get_event_loop()
        dataset = await loop.run_in_executor(None, load_dataset, dataset_name, config)

        # Process each split in the dataset
        for split_name, split_data in dataset.items():
            split_key = f"{config}_{split_name}"
            if split_key in completed_splits:
                logger.info(
                    f"Skipping already completed split: {split_name} in config {config}"
                )
                continue

            try:
                split_result = await process_split_documents(
                    split_data, dataset_name, config, split_name, semaphore
                )

                if split_name in processed_splits:
                    # Concatenate new data to existing split
                    for key in split_result:
                        processed_splits[split_name][key].extend(split_result[key])
                else:
                    processed_splits[split_name] = split_result

                # Mark split as completed
                completed_splits[split_key] = True
                progress_data["completed_splits"] = completed_splits
                save_progress(dataset_name, progress_data)

            except Exception as e:
                logger.error(
                    f"Error processing split {split_name} in config {config}: {e}"
                )
                continue

        # Mark config as completed
        completed_configs.add(config)
        progress_data["completed_configs"] = list(completed_configs)
        save_progress(dataset_name, progress_data)

    # Convert processed splits to Dataset objects
    loop = asyncio.get_event_loop()
    for split_name in processed_splits:
        split_data = processed_splits[split_name]
        logger.info(
            f"Creating Dataset for split '{split_name}' with {len(split_data.get('text', []))} items"
        )
        processed_splits[split_name] = await loop.run_in_executor(
            None, Dataset.from_dict, split_data
        )

    # Create and save final dataset
    dataset_dict = DatasetDict(processed_splits)
    save_final_dataset(dataset_dict, dataset_name)

    return dataset_dict


async def upload_config_to_hf(
    dataset_dict: DatasetDict, repo_id: str, config_name: str, hf_token: str
):
    """
    Uploads a DatasetDict as a specific configuration to HuggingFace Hub asynchronously.
    """
    logger.info(f"Uploading config '{config_name}' to HuggingFace Hub: {repo_id}")

    # Log dataset info before upload
    for split_name, split_dataset in dataset_dict.items():
        logger.info(f"Split '{split_name}': {len(split_dataset)} items")

    # Run the upload in an executor to avoid blocking
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: dataset_dict.push_to_hub(
            repo_id, config_name=config_name, token=hf_token
        ),
    )

    logger.info(f"Upload complete for {repo_id}/{config_name}")


async def process_dataset_async(
    dataset_name: str, repo_id: str, hf_token: str, max_concurrent_requests: int = 5
):
    """Process a single dataset asynchronously with error recovery."""
    try:
        # Process the entire dataset (all configs combined)
        dataset_dict = await process_single_dataset(
            dataset_name, max_concurrent_requests
        )

        # Extract just the dataset name (remove the organization prefix)
        config_name = dataset_name.split("/")[-1]  # e.g., "chempile-reasoning"

        # Upload this dataset as a configuration
        await upload_config_to_hf(dataset_dict, repo_id, config_name, hf_token)

        logger.info(f"Successfully processed and uploaded {dataset_name}")

        # Clean up checkpoints after successful upload (optional)
        # cleanup_checkpoints(dataset_name)

    except Exception as e:
        logger.error(f"Error processing dataset {dataset_name}: {e}")
        logger.info("Progress has been saved. You can resume processing later.")
        raise


def cleanup_checkpoints(dataset_name: str):
    """Clean up checkpoint files for a completed dataset."""
    safe_dataset_name = dataset_name.replace("/", "-")
    for file_path in CHECKPOINT_DIR.glob(f"{safe_dataset_name}*"):
        try:
            file_path.unlink()
            logger.info(f"Cleaned up checkpoint: {file_path}")
        except Exception as e:
            logger.error(f"Failed to clean up {file_path}: {e}")


async def main():
    """Main async function."""
    # List of datasets to process
    datasets = [
        "jablonkagroup/chempile-reasoning",
        "jablonkagroup/chempile-education",
        "jablonkagroup/chempile-lift",
        "jablonkagroup/chempile-code",
        "jablonkagroup/chempile-paper",
    ]

    load_dotenv("../.env", override=True)
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not found.")

    repo_id = "jablonkagroup/chempile-instruction"
    max_concurrent_requests = 20  # Adjust based on your API rate limits

    # Process datasets one by one to be safer with memory and error handling
    for dataset_name in datasets:
        try:
            await process_dataset_async(
                dataset_name, repo_id, hf_token, max_concurrent_requests
            )
        except Exception as e:
            logger.error(f"Failed to process {dataset_name}: {e}")
            logger.info("Continuing with next dataset...")
            continue


if __name__ == "__main__":
    asyncio.run(main())

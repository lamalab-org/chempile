import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv
from datasets import load_dataset, get_dataset_config_names, Dataset, DatasetDict
from pydantic import BaseModel, Field
from typing import Literal, Optional
import pandas as pd
from loguru import logger
import math

load_dotenv("../.env")

client = OpenAI()

SYSTEM_PROMPT = (
    "You are a helpful assistant that rephrases text to make it into multi-turn conversations about chemistry. "
    "Take the provided text and rephrase it to create a conversation that includes user and assistant messages. "
    "There is no limit on the number of turns in the conversation. Make it as rich and informative as possible. "
    "Make the conversation independent of the original text, so it can stand alone as a conversation about chemistry, containing enough context. "
    "The relevant information (e.g., data, molecules) must be in the conversation. Conversations must start with a user request."
)


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class Conversation(BaseModel):
    first_tag: Optional[
        list[
            Literal["requires-knowledge", "requires-calculation", "requires-reasoning"]
        ]
    ] = Field(
        default=None,
        description=(
            "List of possible skill types required for the conversation. "
            "- 'requires-knowledge': Requires factual or domain-specific knowledge. "
            "- 'requires-calculation': Requires mathematical or computational reasoning. "
            "- 'requires-reasoning': Requires logical or deductive reasoning."
        ),
    )
    second_tag: Optional[
        list[
            Literal[
                "Analytical Chemistry",
                "General Chemistry",
                "Inorganic Chemistry",
                "Materials Science",
                "Organic Chemistry",
                "Physical Chemistry",
                "Technical Chemistry",
            ]
        ]
    ] = Field(
        default=None,
        description=(
            "List of possible domains for tagging the conversation.\n"
            "- 'Analytical Chemistry': Focuses on methods and techniques for analyzing substances, such as spectroscopy, chromatography, and mass spectrometry, to identify or quantify chemical compounds.\n"
            "- 'General Chemistry': Covers basic chemistry concepts, including the study of atoms, molecules, reactions, stoichiometry, and the periodic table, suitable for beginners or foundational learning.\n"
            "- 'Inorganic Chemistry': Deals with the properties and behaviors of inorganic compounds, including metals, metal complexes, salts, and minerals, excluding organic chemistry.\n"
            "- 'Materials Science': Involves the study of materials and their properties, including metals, polymers, ceramics, and composites, as well as their applications in various industries.\n"
            "- 'Organic Chemistry': Focuses on the study of carbon-containing compounds, including hydrocarbons and their derivatives, along with the mechanisms, reactions, and synthesis of organic molecules.\n"
            "- 'Physical Chemistry': Explores the principles and theories that explain chemical behavior, including thermodynamics, kinetics, quantum mechanics, and the interaction of matter and energy.\n"
            "- 'Technical Chemistry': Pertains to applied chemical knowledge, focusing on the industrial and practical use of chemical processes, technologies, and techniques in manufacturing and engineering."
        ),
    )
    messages: list[Message]


CONVERSATION_SCHEMA = Conversation.model_json_schema()

HARD_PROMPTS = [
    "Use esoteric vocabulary and complex syntax suitable for academics",
    "Employ arcane terminology and intricate sentence structures for scholarly audiences",
    "Utilize obscure language and convoluted phrasing to challenge comprehension",
    "Incorporate recondite expressions and sophisticated grammar for expert readers",
    "Apply cryptic diction and elaborate constructions to elevate academic rigor",
]

WIKI_PROMPTS = [
    "Use formal, encyclopedic English resembling Wikipedia",
    "Employ objective, authoritative language similar to an encyclopedia entry",
    "Utilize precise, factual phrasing akin to scholarly reference works",
    "Adopt neutral, informative tone characteristic of academic articles",
    "Present information in clear, impersonal style as found in reference materials",
]

ENGAGING_PROMPTS = [
    "Present information in a clear yet engaging manner, suitable for a broad audience",
    "Use an approachable, conversational tone that still maintains a professional and informative style",
    "Employ simple yet precise language with a touch of sophistication, making it both accessible and authoritative",
    "Balance clarity with expert-level knowledge, presenting facts in a digestible and friendly way",
    "Utilize an engaging, direct style that appeals to a wide range of readers while retaining intellectual rigor",
]

MAX_TASKS_PER_FILE = 10000


def prepare_jsonl_files(split_dfs: dict, dset: str):
    """
    Prepare JSONL files, splitting into chunks of MAX_TASKS_PER_FILE tasks each.
    Returns a list of file names.
    """
    import random

    tasks = []
    prompt_sets = [HARD_PROMPTS, WIKI_PROMPTS, ENGAGING_PROMPTS, ""]
    prompt_types = ["hard", "wiki", "engaging", "none"]
    weights = [0.1, 0.3, 0.3, 0.3]

    for split_name, df in split_dfs.items():
        for i, row in df.iterrows():
            text = row["text"]
            chosen_set_idx = random.choices(
                range(len(prompt_sets)), weights=weights, k=1
            )[0]
            chosen_set = prompt_sets[chosen_set_idx]
            prompt_type = prompt_types[chosen_set_idx]

            if isinstance(chosen_set, list):
                user_prompt = random.choice(chosen_set)
            else:
                user_prompt = ""
            prompt_text = f"{user_prompt}. " if user_prompt else ""
            task = {
                "custom_id": f"task-{dset}-{split_name}-{i}-{prompt_type}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini-2024-07-18",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": f"{prompt_text}\nOriginal text:\n{text}\n\nNow, rephrase this text into a multi-turn conversation about chemistry.",
                        },
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "multi_turn_conversation",
                            "schema": CONVERSATION_SCHEMA,
                        },
                    },
                },
            }
            tasks.append(task)

    # Split tasks into chunks
    total_tasks = len(tasks)
    num_files = math.ceil(total_tasks / MAX_TASKS_PER_FILE)

    logger.info(f"Total tasks: {total_tasks}")
    logger.info(
        f"Splitting into {num_files} files of max {MAX_TASKS_PER_FILE} tasks each"
    )

    file_names = []
    for file_idx in range(num_files):
        start_idx = file_idx * MAX_TASKS_PER_FILE
        end_idx = min((file_idx + 1) * MAX_TASKS_PER_FILE, total_tasks)
        chunk_tasks = tasks[start_idx:end_idx]

        f_name = f"{dset}_part_{file_idx + 1}_of_{num_files}.jsonl"
        with open(f_name, "w") as f:
            for task in chunk_tasks:
                f.write(json.dumps(task) + "\n")

        file_names.append(f_name)
        logger.info(f"Created {f_name} with {len(chunk_tasks)} tasks")

    return file_names


def make_batch_calls(file_names: list):
    """
    Submit all batch jobs and return their job IDs and corresponding file names.
    """
    batch_jobs = []

    # Submit all batch jobs
    for f_name in file_names:
        logger.info(f"Submitting batch job for {f_name}")
        with open(f_name, "rb") as f:
            batch_file = client.files.create(file=f, purpose="batch")

        batch_job = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        batch_jobs.append(
            {"job_id": batch_job.id, "input_file": f_name, "status": "submitted"}
        )
        logger.info(f"Submitted batch job {batch_job.id} for {f_name}")

    # Wait for all jobs to complete
    completed_jobs = []
    while len(completed_jobs) < len(batch_jobs):
        for job_info in batch_jobs:
            if job_info["status"] != "completed":
                batch_job = client.batches.retrieve(job_info["job_id"])
                job_info["status"] = batch_job.status

                if batch_job.status == "completed":
                    result_file_id = batch_job.output_file_id
                    logger.info(
                        f"Batch job {job_info['job_id']} completed. Result file ID: {result_file_id}"
                    )

                    result = client.files.content(result_file_id)
                    result = result.content

                    output_file = job_info["input_file"].replace(
                        ".jsonl", "_output.jsonl"
                    )
                    logger.info(f"Saving output to {output_file}")
                    with open(output_file, "wb") as f:
                        f.write(result)
                    logger.info("Output saved successfully.")

                    completed_jobs.append(
                        {
                            "input_file": job_info["input_file"],
                            "output_file": output_file,
                            "job_id": job_info["job_id"],
                        }
                    )

                elif batch_job.status == "failed":
                    logger.error(f"Batch job {job_info['job_id']} failed")
                    raise Exception(f"Batch job {job_info['job_id']} failed")
                elif batch_job.status == "cancelled":
                    logger.error(f"Batch job {job_info['job_id']} was cancelled")
                    raise Exception(f"Batch job {job_info['job_id']} was cancelled")
                elif batch_job.status == "running":
                    logger.info(f"Batch job {job_info['job_id']} is still running...")

        # Show progress
        completed_count = len(completed_jobs)
        total_count = len(batch_jobs)
        logger.info(f"Progress: {completed_count}/{total_count} batch jobs completed")

        if completed_count < total_count:
            logger.info("Waiting for remaining batch jobs to complete...")
            time.sleep(1200)  # Wait 20 minutes before checking again
            logger.info("Checking batch job statuses...")

    return completed_jobs


def concatenate_output_files(completed_jobs: list, dset: str):
    """
    Concatenate all output files into a single file.
    """
    combined_output_file = f"{dset}_combined_output.jsonl"

    logger.info(
        f"Concatenating {len(completed_jobs)} output files into {combined_output_file}"
    )

    with open(combined_output_file, "wb") as outfile:
        for job_info in completed_jobs:
            output_file = job_info["output_file"]
            logger.info(f"Adding content from {output_file}")
            with open(output_file, "rb") as infile:
                outfile.write(infile.read())

    logger.info(f"Combined output saved to {combined_output_file}")
    return combined_output_file


def load_and_save_data(dset):
    split_collections = {}  # Dictionary to hold DataFrames by split
    logger.info(f"Loading dataset {dset}")
    configs = get_dataset_config_names(dset)

    for config in configs:
        logger.info(f"Loading dataset {dset} with config {config}")
        data = load_dataset(dset, config)

        for split_name, split_data in data.items():
            # Convert split_data to pandas DataFrame
            df = pd.DataFrame(split_data)
            if "text" in df.columns:
                # Add metadata columns with original config and split info
                df = df[["text"]].copy()
                df["original_config"] = config
                df["original_dataset"] = dset
                df["original_split"] = split_name

                # Collect DataFrames by split name
                if split_name not in split_collections:
                    split_collections[split_name] = []
                split_collections[split_name].append(df)
            else:
                # If 'text' column is missing, skip or handle as needed
                continue

    # Concatenate DataFrames for each split separately
    final_split_dfs = {}
    for split_name, dfs_list in split_collections.items():
        combined_df = pd.concat(dfs_list, ignore_index=True)
        final_split_dfs[split_name] = combined_df
        logger.info(f"Split '{split_name}': {len(combined_df)} samples")

    return final_split_dfs


# Add this function before the dataset creation part of your code


def validate_and_normalize_data(results, split_name):
    """
    Validate and normalize all data fields to ensure consistent types for PyArrow
    """
    normalized_results = []
    validation_errors = 0

    for i, conv in enumerate(results):
        try:
            # Validate and normalize first_tag
            first_tag = conv.get("first_tag")
            if first_tag is None:
                first_tag = []
            elif isinstance(first_tag, str):
                first_tag = [first_tag]
            elif isinstance(first_tag, list):
                # Ensure all elements are strings
                first_tag = [
                    str(item) if item is not None else "" for item in first_tag
                ]
            else:
                logger.warning(
                    f"Invalid first_tag type for split {split_name}, item {i}: {type(first_tag)}. Converting to empty list."
                )
                first_tag = []

            # Validate and normalize second_tag
            second_tag = conv.get("second_tag")
            if second_tag is None:
                second_tag = []
            elif isinstance(second_tag, str):
                second_tag = [second_tag]
            elif isinstance(second_tag, list):
                # Ensure all elements are strings
                second_tag = [
                    str(item) if item is not None else "" for item in second_tag
                ]
            else:
                logger.warning(
                    f"Invalid second_tag type for split {split_name}, item {i}: {type(second_tag)}. Converting to empty list."
                )
                second_tag = []

            # Validate and normalize origin
            origin = conv.get("origin")
            if origin is None or not isinstance(origin, dict):
                origin = {"dataset": "", "config": "", "split": "", "prompt_type": ""}
            else:
                # Ensure all origin fields are strings
                origin = {
                    "dataset": (
                        str(origin.get("dataset", ""))
                        if origin.get("dataset") is not None
                        else ""
                    ),
                    "config": (
                        str(origin.get("config", ""))
                        if origin.get("config") is not None
                        else ""
                    ),
                    "split": (
                        str(origin.get("split", ""))
                        if origin.get("split") is not None
                        else ""
                    ),
                    "prompt_type": (
                        str(origin.get("prompt_type", ""))
                        if origin.get("prompt_type") is not None
                        else ""
                    ),
                }

            # Validate and normalize messages
            messages = conv.get("messages", [])
            if not isinstance(messages, list):
                logger.warning(
                    f"Messages is not a list for split {split_name}, item {i}. Converting: {type(messages)}"
                )
                messages = []
            else:
                # Normalize each message
                normalized_messages = []
                for j, msg in enumerate(messages):
                    if not isinstance(msg, dict):
                        logger.warning(
                            f"Message {j} is not a dict for split {split_name}, item {i}. Converting: {type(msg)}"
                        )
                        normalized_msg = {
                            "role": "unknown",
                            "content": str(msg) if msg is not None else "",
                        }
                    else:
                        normalized_msg = {
                            "role": (
                                str(msg.get("role", "unknown"))
                                if msg.get("role") is not None
                                else "unknown"
                            ),
                            "content": (
                                str(msg.get("content", ""))
                                if msg.get("content") is not None
                                else ""
                            ),
                        }
                    normalized_messages.append(normalized_msg)
                messages = normalized_messages

            # Create normalized conversation data
            normalized_conv = {
                "first_tag": first_tag,
                "second_tag": second_tag,
                "origin": origin,
                "messages": messages,
            }

            normalized_results.append(normalized_conv)

        except Exception as e:
            logger.error(f"Error validating item {i} in split {split_name}: {e}")
            validation_errors += 1
            continue

    logger.info(
        f"Split '{split_name}': Validated {len(normalized_results)} items, {validation_errors} validation errors"
    )
    return normalized_results


def process_and_upload_data(output_file, dset, hf_token, original_split_dfs):
    # Dictionary to hold results by split
    split_results = {}
    error_count = 0
    success_count = 0

    with open(output_file, "rb") as f:
        for line_num, line in enumerate(f):
            logger.debug(f"Processing line {line_num}")
            try:
                json_object = json.loads(line.strip())
            except Exception as e:
                logger.error(f"Error decoding line {line_num} as JSON: {e}")
                error_count += 1
                continue

            custom_id = json_object.get("custom_id", "")
            logger.debug(f"custom_id: {custom_id}")

            choices = json_object.get("response", {}).get("body", {}).get("choices", [])

            if choices:
                message_content = choices[0].get("message", {}).get("content", "")
                logger.debug(f"Processing message_content for custom_id: {custom_id}")

                try:
                    # Try to parse the JSON content
                    parsed_content = json.loads(message_content)
                    logger.debug(
                        f"Successfully parsed JSON content for custom_id: {custom_id}"
                    )

                    # Validate that parsed_content is a dictionary
                    if not isinstance(parsed_content, dict):
                        logger.error(
                            f"Parsed content is not a dictionary for custom_id: {custom_id}. Type: {type(parsed_content)}"
                        )
                        error_count += 1
                        continue

                    # Handle different key naming conventions (both 'first_tag'/'First Tag' formats)
                    first_tag = parsed_content.get("first_tag") or parsed_content.get(
                        "First Tag"
                    )
                    second_tag = parsed_content.get("second_tag") or parsed_content.get(
                        "Second Tag"
                    )

                    # Ensure first_tag and second_tag are always lists or None
                    if first_tag is None:
                        first_tag = None
                    elif not isinstance(first_tag, list):
                        first_tag = [first_tag]

                    if second_tag is None:
                        second_tag = None
                    elif not isinstance(second_tag, list):
                        second_tag = [second_tag]

                    # Extract origin information
                    try:
                        parts = custom_id.split("-")
                        if len(parts) < 3:
                            raise ValueError(f"custom_id format invalid: {custom_id}")

                        prompt_type = parts[-1]
                        row_index = int(parts[-2])
                        split_name = parts[-3]
                        logger.debug(
                            f"split_name: {split_name}, row_index: {row_index}, prompt_type: {prompt_type}"
                        )

                        if split_name not in original_split_dfs:
                            raise KeyError(
                                f"Split '{split_name}' not found in original_split_dfs"
                            )

                        if row_index >= len(original_split_dfs[split_name]):
                            raise IndexError(
                                f"Row index {row_index} out of bounds for split '{split_name}'"
                            )

                        original_row = original_split_dfs[split_name].iloc[row_index]
                        origin = {
                            "dataset": original_row["original_dataset"],
                            "config": original_row["original_config"],
                            "split": original_row["original_split"],
                            "prompt_type": prompt_type,
                        }
                    except (ValueError, IndexError, KeyError) as e:
                        logger.warning(
                            f"Could not extract origin for custom_id: {custom_id}, error: {e}"
                        )
                        origin = {
                            "dataset": None,
                            "config": None,
                            "split": None,
                            "prompt_type": None,
                        }  # Use consistent dict structure instead of None
                        split_name = "unknown"

                    # Extract and validate messages (handle different key naming conventions)
                    messages = parsed_content.get("messages", []) or parsed_content.get(
                        "Messages", []
                    )

                    # Detailed validation of messages
                    if not isinstance(messages, list):
                        logger.error(
                            f"Messages field is not a list for custom_id: {custom_id}. Type: {type(messages)}, Value: {str(messages)[:200]}..."
                        )
                        error_count += 1
                        continue

                    if not messages:
                        logger.error(
                            f"No messages found for custom_id: {custom_id} at line {line_num}"
                        )
                        logger.error(
                            f"Parsed content keys: {list(parsed_content.keys())}"
                        )
                        logger.error(
                            f"Full parsed content (first 500 chars): {str(parsed_content)[:500]}..."
                        )
                        error_count += 1
                        continue

                    # Validate each message in the list
                    valid_messages = []
                    for i, msg in enumerate(messages):
                        if not isinstance(msg, dict):
                            logger.warning(
                                f"Message {i} is not a dict for custom_id: {custom_id}. Converting: {msg}"
                            )
                            # Try to convert non-dict messages to a reasonable format
                            valid_messages.append(
                                {"role": "unknown", "content": str(msg)}
                            )
                        else:
                            valid_messages.append(
                                {
                                    "role": msg.get("role", "unknown"),
                                    "content": msg.get("content", ""),
                                }
                            )

                    conversation_data = {
                        "first_tag": first_tag,
                        "second_tag": second_tag,
                        "origin": origin,
                        "messages": valid_messages,
                    }

                    logger.debug(
                        f"Successfully processed conversation_data for split {split_name}"
                    )

                    if split_name not in split_results:
                        split_results[split_name] = []
                    split_results[split_name].append(conversation_data)
                    success_count += 1

                except json.JSONDecodeError as e:
                    logger.error(
                        f"JSON decode error for custom_id: {custom_id} at line {line_num}"
                    )
                    logger.error(f"JSON error: {e}")
                    logger.error(
                        f"Problematic content (first 300 chars): {message_content[:300]}..."
                    )
                    # Try to identify common JSON issues
                    if "'" in message_content and '"' not in message_content[:50]:
                        logger.error(
                            "Possible issue: Single quotes instead of double quotes in JSON"
                        )
                    if message_content.count("{") != message_content.count("}"):
                        logger.error("Possible issue: Mismatched braces")
                    if message_content.count("[") != message_content.count("]"):
                        logger.error("Possible issue: Mismatched brackets")
                    error_count += 1
                    continue

                except Exception as e:
                    logger.error(
                        f"Unexpected error processing custom_id: {custom_id} at line {line_num}"
                    )
                    logger.error(f"Error: {e}")
                    logger.error(
                        f"Message content (first 300 chars): {message_content[:300]}..."
                    )
                    if "parsed_content" in locals():
                        logger.error(f"Parsed content type: {type(parsed_content)}")
                        logger.error(
                            f"Parsed content (first 300 chars): {str(parsed_content)[:300]}..."
                        )
                    error_count += 1
                    continue

    # Log processing summary
    logger.info(f"Processing complete. Success: {success_count}, Errors: {error_count}")

    # Convert each split's results into Hugging Face Datasets
    datasets_by_split = {}
    for split_name, results in split_results.items():
        if results:  # Only create dataset if there are results
            logger.info(f"Validating and normalizing data for split '{split_name}'...")

            # Validate and normalize all data
            normalized_results = validate_and_normalize_data(results, split_name)

            if not normalized_results:
                logger.warning(
                    f"No valid data after normalization for split '{split_name}'"
                )
                continue

            # Extract data for dataset creation with additional type checking
            try:
                first_tags = [conv["first_tag"] for conv in normalized_results]
                second_tags = [conv["second_tag"] for conv in normalized_results]
                origins = [conv["origin"] for conv in normalized_results]
                messages_list = [conv["messages"] for conv in normalized_results]

                # Final type validation before dataset creation
                logger.info(f"Final validation for split '{split_name}':")
                logger.info(
                    f"  first_tags: all lists? {all(isinstance(x, list) for x in first_tags)}"
                )
                logger.info(
                    f"  second_tags: all lists? {all(isinstance(x, list) for x in second_tags)}"
                )
                logger.info(
                    f"  origins: all dicts? {all(isinstance(x, dict) for x in origins)}"
                )
                logger.info(
                    f"  messages: all lists? {all(isinstance(x, list) for x in messages_list)}"
                )

                # Check for any remaining None values that could cause issues
                none_check_fields = [
                    ("first_tags", first_tags),
                    ("second_tags", second_tags),
                    ("origins", origins),
                    ("messages", messages_list),
                ]

                for field_name, field_data in none_check_fields:
                    none_count = sum(1 for x in field_data if x is None)
                    if none_count > 0:
                        logger.warning(
                            f"Found {none_count} None values in {field_name}"
                        )

                split_dataset = Dataset.from_dict(
                    {
                        "first_tag": first_tags,
                        "second_tag": second_tags,
                        "origin": origins,
                        "messages": messages_list,
                    }
                )

                datasets_by_split[split_name] = split_dataset
                logger.info(
                    f"Split '{split_name}': {len(split_dataset)} samples successfully created"
                )

            except Exception as e:
                logger.error(f"Error creating dataset for split '{split_name}': {e}")
                logger.error("Sample data types:")
                if normalized_results:
                    sample = normalized_results[0]
                    for key, value in sample.items():
                        logger.error(f"  {key}: {type(value)} = {str(value)[:100]}...")
                continue

    if not datasets_by_split:
        logger.error("No valid datasets created. Check the input data format.")
        return False

    # Create DatasetDict with preserved splits
    dataset_dict = DatasetDict(datasets_by_split)

    # Log the final split sizes
    for split_name, dataset in dataset_dict.items():
        logger.info(f"Final {split_name} samples: {len(dataset)}")

    # Upload to Hugging Face Hub
    try:
        dataset_dict.push_to_hub(
            "jablonkagroup/chempile-instruction",
            config_name=dset,
            token=hf_token,
            private=False,  # Set to True if you want a private dataset
        )
        logger.info(
            "Dataset uploaded successfully to jablonkagroup/chempile-instruction"
        )
        return True
    except Exception as e:
        logger.error(f"Error uploading dataset: {e}")
        return False


def main():
    """
    Main function to load the dataset, prepare the JSONL files,
    make batch calls to the OpenAI API, process the results,
    and upload the final dataset to the Hugging Face Hub.
    """
    dataset = "chempile-reasoning"
    full_dset_name = f"jablonkagroup/{dataset}"
    # Load environment variables
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise ValueError(
            "Hugging Face token not found. Please add it to the ../.env file."
        )

    split_dfs = load_and_save_data(full_dset_name)

    # Prepare JSONL files (potentially multiple files)
    input_files = prepare_jsonl_files(split_dfs, dataset)

    # Submit batch jobs and wait for completion
    completed_jobs = make_batch_calls(input_files)

    # Concatenate all output files
    combined_output_file = concatenate_output_files(completed_jobs, dataset)

    # Process the combined output file
    success = process_and_upload_data(
        combined_output_file, dataset, hf_token, split_dfs
    )

    if success:
        logger.info("Data processing and upload completed successfully.")
    else:
        logger.error("Data processing or upload failed.")


if __name__ == "__main__":
    main()

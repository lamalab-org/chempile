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


def prepare_jsonl_file(df: pd.DataFrame, dset: str):
    import random

    tasks = []
    prompt_sets = [HARD_PROMPTS, WIKI_PROMPTS, ENGAGING_PROMPTS, ""]
    weights = [0.1, 0.3, 0.3, 0.3]
    for i, row in df.iterrows():
        text = row["text"]
        chosen_set = random.choices(prompt_sets, weights=weights, k=1)[0]
        if isinstance(chosen_set, list):
            user_prompt = random.choice(chosen_set)
        else:
            user_prompt = ""
        prompt_text = f"{user_prompt}. " if user_prompt else ""
        task = {
            "custom_id": f"task-{dset}-{i}",
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
    f_name = f"{dset}.jsonl"
    with open(f_name, "w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")

    return f_name


def make_batch_call(f_name: str):
    # Open file in binary mode for upload
    with open(f_name, "rb") as f:
        batch_file = client.files.create(file=f, purpose="batch")
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    batch_job = client.batches.retrieve(batch_job.id)
    while batch_job.status != "completed":
        batch_job = client.batches.retrieve(batch_job.id)
        logger.info(f"Batch job status: {batch_job.status}")
        if batch_job.status == "failed":
            logger.error("Batch job failed")
            raise Exception("Batch job failed")
        elif batch_job.status == "running":
            logger.info("Batch job is still running...")
        elif batch_job.status == "cancelled":
            logger.error("Batch job was cancelled")
        logger.info("Waiting for batch job to complete...")
        time.sleep(300)  # Wait
        logger.info("Checking batch job status...")

    result_file_id = batch_job.output_file_id
    logger.info(f"Batch job completed. Result file ID: {result_file_id}")

    result = client.files.content(result_file_id)
    result = result.content

    output_file = f_name.replace(".jsonl", "_output.jsonl")
    logger.info(f"Saving output to {output_file}")
    with open(output_file, "wb") as f:
        f.write(result)
    logger.info("Output saved successfully.")

    return output_file


def load_and_save_data(dset):
    all_rows = []
    logger.info(f"Loading dataset {dset}")
    configs = get_dataset_config_names(dset)
    for config in configs:
        if config != "spectra_reasoning_deepseek-default":
            logger.info(f"Loading dataset {dset} with config {config}")
            continue
        logger.info(f"Loading dataset {dset} with config {config}")
        data = load_dataset(dset, config)
        for split_name, split_data in data.items():
            # Convert split_data to pandas DataFrame and collect rows
            df = pd.DataFrame(split_data)
            if "text" in df.columns:
                # Add metadata column with original config and split info
                df = df[["text"]].copy()
                df["original_config"] = config
                df["original_dataset"] = dset
                df["original_split"] = split_name
                all_rows.append(df)
            else:
                # If 'text' column is missing, skip or handle as needed
                continue
    # Concatenate all DataFrames and reset index to ensure continuous indexing
    big_df = pd.concat(all_rows, ignore_index=True)
    return big_df


def process_and_upload_data(output_file, dset, hf_token, original_df):
    results = []

    with open(output_file, "rb") as f:
        for line in f:
            # Parse the JSON string into a Python dictionary
            json_object = json.loads(line.strip())

            # Extract custom_id to get the original row index
            custom_id = json_object.get("custom_id", "")

            # Extract the relevant part from the JSON object
            choices = json_object.get("response", {}).get("body", {}).get("choices", [])

            if choices:
                message_content = choices[0].get("message", {}).get("content", "")

                # Parse the message content (it is a stringified JSON)
                try:
                    parsed_content = json.loads(message_content)

                    first_tag = parsed_content.get("first_tag") or None
                    second_tag = parsed_content.get("second_tag") or None

                    # Extract row index from custom_id (format: "task-{dset}-{i}")
                    try:
                        row_index = int(custom_id.split("-")[-1])
                        original_row = original_df.iloc[row_index]
                        origin = {
                            "dataset": original_row["original_dataset"],
                            "config": original_row["original_config"],
                            "split": original_row["original_split"],
                        }
                    except (ValueError, IndexError, KeyError):
                        logger.warning(
                            f"Could not extract origin for custom_id: {custom_id}"
                        )
                        origin = None

                    # Create the conversation using the parsed content
                    conversation_data = {
                        "first_tag": first_tag,
                        "second_tag": second_tag,
                        "origin": origin,
                        "messages": [
                            {
                                "role": msg.get("role", ""),
                                "content": msg.get("content", ""),
                            }
                            for msg in parsed_content.get("messages", [])
                        ],
                    }

                    # Attempt to create a Conversation instance from the parsed data
                    # Note: We don't validate with Pydantic here since we're adding the origin field
                    results.append(conversation_data)
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding content JSON: {e}")
                except Exception as e:
                    logger.error(f"Error processing conversation data: {e}")

    # Convert the results list into a Hugging Face Dataset
    full_dataset = Dataset.from_dict(
        {
            "first_tag": [conv["first_tag"] for conv in results],
            "second_tag": [conv["second_tag"] for conv in results],
            "origin": [conv["origin"] for conv in results],
            "messages": [conv["messages"] for conv in results],
        }
    )

    # Split the dataset into train/val/test (90/5/5)
    logger.info(f"Total samples: {len(full_dataset)}")

    # First split: 90% train, 10% temp (for val+test)
    train_test_split = full_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split["train"]
    temp_dataset = train_test_split["test"]

    # Second split: split the 10% into 5% val and 5% test
    val_test_split = temp_dataset.train_test_split(test_size=0.5, seed=42)
    val_dataset = val_test_split["train"]
    test_dataset = val_test_split["test"]

    # Create DatasetDict with splits
    dataset_dict = DatasetDict(
        {"train": train_dataset, "val": val_dataset, "test": test_dataset}
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

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
    Main function to create, combine, and upload the dataset.

    This function orchestrates the sampling of chemistry and Gutenberg data,
    combines them, shuffles the resulting dataset, and uploads it to the
    Hugging Face Hub.
    """
    dataset = "chempile-reasoning"
    full_dset_name = f"jablonkagroup/{dataset}"
    # Load environment variables
    load_dotenv("../.env")
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise ValueError(
            "Hugging Face token not found. Please add it to the ../.env file."
        )

    df = load_and_save_data(full_dset_name)
    input_file = prepare_jsonl_file(df, dataset)
    output_file = make_batch_call(input_file)
    success = process_and_upload_data(output_file, dataset, hf_token, df)
    if success:
        logger.info("Data processing and upload completed successfully.")
    else:
        logger.error("Data processing or upload failed.")


if __name__ == "__main__":
    main()

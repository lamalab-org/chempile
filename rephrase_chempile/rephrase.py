from dotenv import load_dotenv
from openai import OpenAI
from datasets import get_dataset_config_names, load_dataset, Dataset, DatasetDict
import random
import tiktoken
import os

load_dotenv("../.env", override=True)

client = OpenAI()


SIMPLE_PROMPT = """Given the text below:

{document}

Generate exactly one question and its corresponding answer based solely on the content.

Format your response as:
Question: [your question]
Answer: [your answer]"""

HARD_PROMPT = """Rephrase the text below into a scholarly dialogue using these requirements:

1. Structure as multiple alternating "Question:" and "Answer:" pairs
2. Use esoteric vocabulary and complex syntax suitable for academics
3. Replace common terms with rare, technical alternatives

Text:

{document}

Provide only the reformatted dialogue:"""

WIKI_PROMPT = """Convert the text below into a dialogue meeting these criteria:

1. Structure as multiple "Question:" and "Answer:" pairs
2. Use formal, encyclopedic English resembling Wikipedia
3. Maintain factual accuracy and neutral tone

Text:

{document}

Output only the formatted dialogue:"""

BASE_QA_PROMPT = """Transform the text below into dialogue format using:

- Multiple alternating "Question:" and "Answer:" pairs

Text:

{document}

Provide only the resulting dialogue:"""


def make_call(prompt):
    response = client.responses.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
    )
    return response.output_text


def check_configs(dataset_name: str):
    try:
        configs = get_dataset_config_names(dataset_name)
        return configs
    except Exception as e:
        print(f"Error retrieving configs for {dataset_name}: {e}")
        return []


def extract_question_answer(response: str):
    """
    Extracts the question and answer from the response string.
    Handles multi-line questions and answers.
    Assumes the response is formatted as:
    Question: [question text possibly with newlines]
    Answer: [answer text possibly with newlines]
    """
    response = response.strip()
    q_idx = response.find("Question:")
    a_idx = response.find("Answer:")
    if q_idx == -1 or a_idx == -1:
        print("Warning: Could not find 'Question:' or 'Answer:' in response.")
        return None
    question = response[q_idx + len("Question:") : a_idx].strip()
    answer = response[a_idx + len("Answer:") :].strip()
    if question and answer:
        return {"question": question, "answer": answer}
    else:
        print("Warning: Could not extract question and answer text.")
        return None


def rephrase(document: str):

    # Count tokens and truncate if necessary
    max_tokens = 120000
    num_tokens = count_tokens(document)
    if num_tokens > max_tokens:
        print(f"Document has {num_tokens} tokens, truncating to {max_tokens} tokens.")
        encoding = tiktoken.encoding_for_model("gpt2")
        tokens = encoding.encode(document)
        truncated_tokens = tokens[:max_tokens]
        document = encoding.decode(truncated_tokens)
    else:
        print(f"Document has {num_tokens} tokens, no truncation needed.")

    prompts = [SIMPLE_PROMPT, HARD_PROMPT, WIKI_PROMPT, BASE_QA_PROMPT]
    weights = [0.3, 0.1, 0.3, 0.3]  # Assigning less weight to HARD_PROMPT
    prompt = random.choices(prompts, weights=weights, k=1)[0]
    formatted_prompt = prompt.format(document=document)
    print(f"Rephrasing document with prompt:\n{formatted_prompt}")
    response = make_call(formatted_prompt)
    return extract_question_answer(response)


def count_tokens(text: str, model: str = "gpt2") -> int:
    """
    Counts the number of tokens in a given text using the specified model's tokenizer.

    Args:
        text (str): The input string to tokenize.
        model (str): The OpenAI model name (e.g., "gpt-4", "gpt-3.5-turbo", "text-davinci-003").

    Returns:
        int: The number of tokens in the input text.
    """
    try:
        # Get encoding for the model
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback if the model is unknown
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)
    return len(tokens)


def process_single_dataset(dataset_name: str):
    """
    Process a single dataset and return a DatasetDict with all its splits combined across configs.
    """
    print(f"Processing dataset: {dataset_name}")

    # Get all configs for this dataset
    configs = check_configs(dataset_name)
    if not configs:
        raise ValueError(f"No configurations found for dataset: {dataset_name}")

    # Dictionary to store processed splits for this dataset
    processed_splits = {}

    for config in configs:
        print(f"Processing config: {config}")
        # Load the dataset with the specific config
        dataset = load_dataset(dataset_name, config)

        # Process each split in the dataset
        for split_name, split_data in dataset.items():
            print(f"Processing split: {split_name}")

            # Prepare data for Dataset creation
            texts = []
            questions = []
            answers = []
            metadata = []

            for idx, document in enumerate(split_data):
                rephrased_results = []
                rephrased_results = rephrase(document["text"])

                for result in rephrased_results:
                    if result is None:
                        print(
                            f"Warning: Failed to extract question/answer for document index {idx} in split '{split_name}'"
                        )
                    else:
                        questions.append(result.get("question", ""))
                        answers.append(result.get("answer", ""))
                        texts.append(document["text"])
                        metadata.append(
                            {
                                "original_dataset": dataset_name,
                                "original_config": config,
                                "original_split": split_name,
                                "original_index": idx,
                            }
                        )

            # If this split already exists, append to it (combining across configs)
            if split_name in processed_splits:
                # Concatenate new data to existing split
                processed_splits[split_name]["text"].extend(texts)
                processed_splits[split_name]["question"].extend(questions)
                processed_splits[split_name]["answer"].extend(answers)
                processed_splits[split_name]["metadata"].extend(metadata)
            else:
                processed_splits[split_name] = {
                    "text": texts,
                    "question": questions,
                    "answer": answers,
                    "metadata": metadata,
                }

    # Convert processed splits to Dataset objects
    for split_name in processed_splits:
        split_data = processed_splits[split_name]
        processed_splits[split_name] = Dataset.from_dict(split_data)

    # Return DatasetDict for this dataset
    return DatasetDict(processed_splits)


def upload_config_to_hf(dataset_dict, repo_id, config_name, hf_token):
    """
    Uploads a DatasetDict as a specific configuration to HuggingFace Hub.
    """
    print(f"Uploading config '{config_name}' to HuggingFace Hub: {repo_id}")
    dataset_dict.push_to_hub(repo_id, config_name=config_name, token=hf_token)
    print(f"Upload complete for {repo_id}/{config_name}")


def main():
    # List of datasets to process
    datasets = [
        "jablonkagroup/chempile-reasoning",
        "jablonkagroup/chempile-education",
        "jablonkagroup/chempile-lift-merged",
        "jablonkagroup/chempile-code",
        "jablonkagroup/chempile-paper",
    ]

    load_dotenv("../.env", override=True)
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not found.")

    repo_id = "MrtinoRG/test-instruction"

    # Process each dataset and upload as a separate configuration
    for dataset_name in datasets:
        # Process the entire dataset (all configs combined)
        dataset_dict = process_single_dataset(dataset_name)

        # Extract just the dataset name (remove the organization prefix)
        config_name = dataset_name.split("/")[-1]  # e.g., "chempile-reasoning"

        # Upload this dataset as a configuration
        upload_config_to_hf(dataset_dict, repo_id, config_name, hf_token)


if __name__ == "__main__":
    main()

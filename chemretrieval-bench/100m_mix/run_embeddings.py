# Install required packages (uncomment if running in a fresh environment)
# !pip install scikit-learn datasets openai python-dotenv numpy pandas

from datasets import load_dataset
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
from typing import List
from loguru import logger
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
from accelerate import PartialState

load_dotenv("../.env", override=True)
# Initialize the model and tokenizer
# Configure 4-bit quantization for memory efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load base model with quantization
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb_config,
    device_map={"": PartialState().process_index},
    torch_dtype=torch.float16,
)

# Prepare for LoRA
base_model = prepare_model_for_kbit_training(base_model)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model, "jablonkagroup/chempile-mix-ckpt-checkpoint-100"
)
model_name = "jablonkagroup/chempile-mix-ckpt-checkpoint-100"

# ============================================================================
# ORIGINAL CHEMHOTPOTQA RETRIEVAL CODE
# ============================================================================

# Load datasets
ds_core = load_dataset("BASF-AI/ChemHotpotQARetrieval", "default")
ds_corpus = load_dataset("BASF-AI/ChemHotpotQARetrieval", "corpus")
ds_queries = load_dataset("BASF-AI/ChemHotpotQARetrieval", "queries")


def get_embedding(text):
    """
    Generate embeddings by averaging the last hidden states of all tokens
    from a HuggingFace causal language model.

    Args:
        text (str): Input text to embed

    Returns:
        list: Averaged embedding vector
    """
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)

    # Move inputs to the same device as the model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate with the model and get hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Get the last hidden states (last layer)
    # Shape: (batch_size, sequence_length, hidden_size)
    last_hidden_states = outputs.hidden_states[-1]

    # Average over all tokens (sequence dimension)
    # Shape: (batch_size, hidden_size) -> (hidden_size,)
    averaged_embedding = last_hidden_states.mean(dim=1).squeeze(0)

    # Convert to list and return
    return averaged_embedding.cpu().tolist()


def save_embeddings(embeddings, filename):
    np.save(filename, embeddings)


# Generate and save corpus embeddings if not already present
corpus_embeddings_path = "corpus_embeddings.npy"
if not os.path.exists(corpus_embeddings_path):
    corpus_embeddings = [get_embedding(text) for text in ds_corpus["corpus"]["text"]]
    save_embeddings(corpus_embeddings, corpus_embeddings_path)
else:
    logger.info(f"{corpus_embeddings_path} already exists. Skipping generation.")

# Generate and save queries embeddings if not already present
queries_embeddings_path = "queries_embeddings.npy"
if not os.path.exists(queries_embeddings_path):
    queries_embeddings = [get_embedding(text) for text in ds_queries["queries"]["text"]]
    save_embeddings(queries_embeddings, queries_embeddings_path)
else:
    logger.info(f"{queries_embeddings_path} already exists. Skipping generation.")

# Load embeddings from file
corpus_embeddings = np.load("corpus_embeddings.npy")
queries_embeddings = np.load("queries_embeddings.npy")


def compute_dcg_at_k(relevances: List[float], k: int) -> float:
    """
    Compute Discounted Cumulative Gain at rank k.
    Args:
        relevances: List of relevance scores for ranked items
        k: Rank cutoff
    Returns:
        DCG@k score
    """
    relevances = np.array(relevances[:k])
    if len(relevances) == 0:
        return 0.0
    ranks = np.arange(1, len(relevances) + 1)
    dcg = np.sum((2**relevances - 1) / np.log2(ranks + 1))
    return dcg


def compute_ndcg_at_k(relevances: List[float], k: int) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at rank k.
    Args:
        relevances: List of relevance scores for ranked items
        k: Rank cutoff
    Returns:
        NDCG@k score
    """
    dcg = compute_dcg_at_k(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = compute_dcg_at_k(ideal_relevances, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


# Ensure embeddings are numpy arrays
queries_embeddings = np.array(queries_embeddings)
corpus_embeddings = np.array(corpus_embeddings)

# Note: Need to check how they use train/valid/test splits. Perhaps the loop over the corpus should be changed.

# Non-vectorized evaluation loop
results = []
corpus_df = ds_corpus["corpus"].to_pandas()
k = 10
corpus_ids = corpus_df["_id"].tolist()
for i, row in ds_core["train"].to_pandas().iterrows():
    query = row["query-id"]
    ground_truth = row["corpus-id"]
    ground_truths = []
    similarities = []
    for j, corpus_row in corpus_df.iterrows():
        similarity = cosine_similarity([queries_embeddings[i]], [corpus_embeddings[j]])[
            0
        ][0]
        similarities.append(similarity)
        ground_truths.append(1 if corpus_row["_id"] == ground_truth else 0)
    top_indices = np.argsort(similarities)[::-1][:k]
    top_corpus_ids = [corpus_ids[idx] for idx in top_indices]
    relevant_corpus_ids = {ground_truth}
    relevances = [
        1.0 if corpus_id in relevant_corpus_ids else 0.0 for corpus_id in top_corpus_ids
    ]
    results.append(
        {
            "query-id": query,
            "ndcg_at_10": compute_ndcg_at_k(relevances, k),
        }
    )


def compute_ndcg_fully_vectorized(
    queries_embeddings, corpus_embeddings, ds_core, ds_corpus, k=10
):
    """
    Fully vectorized computation - processes all queries simultaneously.
    """
    train_df = ds_core["train"].to_pandas()
    corpus_df = ds_corpus["corpus"].to_pandas()
    corpus_ids = np.array(corpus_df["_id"].tolist())
    num_queries_embeddings = len(queries_embeddings)
    num_queries_df = len(train_df)
    num_queries = min(num_queries_embeddings, num_queries_df)
    print(
        f"Processing {num_queries} queries (embeddings: {num_queries_embeddings}, df: {num_queries_df})"
    )
    aligned_embeddings = queries_embeddings[:num_queries]
    aligned_df = train_df.iloc[:num_queries]
    similarities_matrix = cosine_similarity(aligned_embeddings, corpus_embeddings)
    top_k_indices = np.argsort(similarities_matrix, axis=1)[:, ::-1][:, :k]
    query_ground_truths = aligned_df["corpus-id"].values
    top_k_corpus_ids = corpus_ids[top_k_indices]
    relevances_matrix = (top_k_corpus_ids == query_ground_truths[:, np.newaxis]).astype(
        float
    )
    ndcg_scores = []
    for i in range(num_queries):
        relevances = relevances_matrix[i]
        ndcg_scores.append(compute_ndcg_at_k(relevances.tolist(), k))
    results = []
    for idx, (_, row) in enumerate(aligned_df.iterrows()):
        results.append(
            {
                "query-id": row["query-id"],
                "ndcg_at_10": ndcg_scores[idx],
            }
        )
    return results


results_frame = pd.DataFrame(
    compute_ndcg_fully_vectorized(
        queries_embeddings, corpus_embeddings, ds_core, ds_corpus
    )
)

mean_ndcg_10 = results_frame["ndcg_at_10"].mean()
logger.debug(f"CHEMHOTPOTQA RETRIEVAL RESULTS: Mean NDCG@10: {mean_ndcg_10}")

# ============================================================================
# NEW CHEMEMB CLASSIFICATION CODE
# ============================================================================

logger.debug("Loading ChemEmb dataset...")
ds_chememb = load_dataset("jablonkagroup/ChemEmb")

# Generate embeddings for ChemEmb test texts
logger.debug("Generating embeddings for ChemEmb test texts...")
chememb_test_df = ds_chememb["test"].to_pandas()
chememb_texts = chememb_test_df["text"].tolist()
chememb_labels = chememb_test_df["chemistry_content"].tolist()

chememb_text_embeddings_path = "chememb_text_embeddings.npy"
chememb_query_embedding_path = "chememb_query_embedding.npy"

# Generate and save ChemEmb text embeddings if not already present
if not os.path.exists(chememb_text_embeddings_path):
    chememb_text_embeddings = [get_embedding(text) for text in chememb_texts]
    save_embeddings(chememb_text_embeddings, chememb_text_embeddings_path)
else:
    logger.info(f"{chememb_text_embeddings_path} already exists. Skipping generation.")

# Generate and save ChemEmb query embedding if not already present
classification_query = "Is this text about chemistry? Yes or No?"
if not os.path.exists(chememb_query_embedding_path):
    query_embedding = get_embedding(classification_query)
    save_embeddings([query_embedding], chememb_query_embedding_path)
else:
    logger.info(f"{chememb_query_embedding_path} already exists. Skipping generation.")

# Load embeddings from file
chememb_text_embeddings = np.load("chememb_text_embeddings.npy")
chememb_query_embedding = np.load("chememb_query_embedding.npy")[0]

logger.debug(f"Processing {len(chememb_texts)} texts for chemistry classification...")


def classify_chemistry_content_vectorized(
    query_embedding, text_embeddings, threshold=0.0
):
    """
    Classify texts as chemistry-related based on cosine similarity with query.
    Higher similarity means more likely to be about chemistry.

    Args:
        query_embedding: Embedding of the classification query
        text_embeddings: Embeddings of texts to classify
        threshold: Similarity threshold for classification (default: 0.5)

    Returns:
        predictions: Binary predictions (1 for chemistry, 0 for not chemistry)
        similarities: Cosine similarity scores
    """
    # Compute similarities between query and all texts
    similarities = cosine_similarity([query_embedding], text_embeddings)[0]

    # Classify based on threshold
    predictions = (similarities > threshold).astype(int)

    return predictions, similarities


# Perform classification
predictions, similarities = classify_chemistry_content_vectorized(
    chememb_query_embedding, chememb_text_embeddings, threshold=0.0
)

# Convert boolean labels to int for consistency
true_labels = [int(label) for label in chememb_labels]

# Calculate metrics
accuracy = accuracy_score(true_labels, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(
    true_labels, predictions, average="binary"
)

# Create results dataframe
chememb_results = []
for i, (text, true_label, pred, sim) in enumerate(
    zip(chememb_texts, true_labels, predictions, similarities)
):
    chememb_results.append(
        {
            "text_id": i,
            "true_label": true_label,
            "predicted_label": pred,
            "similarity_score": sim,
            "correct": true_label == pred,
        }
    )

chememb_results_df = pd.DataFrame(chememb_results)

logger.debug(
    f"CHEMEMB CLASSIFICATION RESULTS: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}"
)

logger.debug("Detailed Classification Report:")
logger.debug(
    classification_report(
        true_labels, predictions, target_names=["Not Chemistry", "Chemistry"]
    )
)

# Analyze threshold sensitivity

# Write results to file
import json

output_filename = "model_scores.json"
scores = {
    "model": model_name,
    "mean_ndcg_10": round(mean_ndcg_10, 4),
    "chememb_classification_accuracy": round(accuracy, 4),
}
with open(output_filename, "w") as f:
    json.dump(scores, f, indent=2)

logger.debug(f"Results written to {output_filename}")

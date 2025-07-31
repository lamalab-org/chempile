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
from sentence_transformers import SentenceTransformer
import torch
import gc
import psutil
import math
import json


# Memory monitoring function
def print_memory_usage(stage=""):
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
    print(f"Memory usage {stage}: {memory_gb:.2f} GB")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i} - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")


def main():
    print_memory_usage("Initial")

    # Check available GPUs
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    model_name = "Qwen/Qwen3-Embedding-8B"

    print_memory_usage("Before model loading")

    # Load environment variables
    load_dotenv()

    # ============================================================================
    # MEMORY-OPTIMIZED DATASET LOADING
    # ============================================================================

    print("Loading datasets...")
    ds_core = load_dataset("BASF-AI/ChemHotpotQARetrieval", "default")
    ds_corpus = load_dataset("BASF-AI/ChemHotpotQARetrieval", "corpus")
    ds_queries = load_dataset("BASF-AI/ChemHotpotQARetrieval", "queries")

    print_memory_usage("After dataset loading")

    def get_embeddings_multi_gpu_sequential(texts, is_query=False, batch_size=4):
        """
        Sequential multi-GPU processing - load model on each GPU one at a time.
        This avoids multiprocessing issues while still using multiple GPUs.
        """
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        total_texts = len(texts)
        print(
            f"Processing {total_texts} texts across {num_gpus} GPUs with batch size {batch_size}"
        )

        # Split texts across GPUs
        texts_per_gpu = math.ceil(total_texts / num_gpus)
        all_embeddings = []

        for gpu_id in range(num_gpus):
            start_idx = gpu_id * texts_per_gpu
            end_idx = min(start_idx + texts_per_gpu, total_texts)

            if start_idx >= total_texts:
                break

            gpu_texts = texts[start_idx:end_idx]
            print(
                f"GPU {gpu_id}: Processing texts {start_idx}-{end_idx} ({len(gpu_texts)} texts)"
            )

            # Load model on specific GPU
            device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
            model = SentenceTransformer(model_name, device=device)

            try:
                # Process in small batches
                gpu_embeddings = []
                for i in range(0, len(gpu_texts), batch_size):
                    batch_end = min(i + batch_size, len(gpu_texts))
                    batch = gpu_texts[i:batch_end]

                    if i % (batch_size * 10) == 0:  # Progress every 10 batches
                        print(
                            f"  GPU {gpu_id}: Batch {i//batch_size + 1}/{math.ceil(len(gpu_texts)/batch_size)}"
                        )
                        print_memory_usage(f"GPU {gpu_id} batch {i}")

                    if is_query:
                        batch_embeddings = model.encode(
                            batch,
                            prompt_name="query",
                            batch_size=batch_size,
                            show_progress_bar=False,
                        )
                    else:
                        batch_embeddings = model.encode(
                            batch, batch_size=batch_size, show_progress_bar=False
                        )

                    gpu_embeddings.extend(batch_embeddings)

                    # Periodic cleanup
                    if i % (batch_size * 20) == 0:
                        gc.collect()
                        torch.cuda.empty_cache()

                all_embeddings.extend(gpu_embeddings)
                print(f"GPU {gpu_id}: Completed {len(gpu_embeddings)} embeddings")

            except Exception as e:
                print(f"Error on GPU {gpu_id}: {e}")
                # Fall back to CPU for this chunk
                print(f"Falling back to CPU for GPU {gpu_id} chunk")
                model = SentenceTransformer(model_name, device="cpu")
                if is_query:
                    fallback_embeddings = model.encode(
                        gpu_texts,
                        prompt_name="query",
                        batch_size=1,
                        show_progress_bar=True,
                    )
                else:
                    fallback_embeddings = model.encode(
                        gpu_texts, batch_size=1, show_progress_bar=True
                    )
                all_embeddings.extend(fallback_embeddings)

            finally:
                # Clean up model and GPU memory
                del model
                gc.collect()
                torch.cuda.empty_cache()
                print_memory_usage(f"After GPU {gpu_id} cleanup")

        return np.array(all_embeddings)

    def get_embeddings_single_gpu_chunked(
        texts, is_query=False, batch_size=4, chunk_size=500
    ):
        """
        Single GPU processing with chunked saving to avoid memory buildup.
        """
        total_texts = len(texts)
        print(
            f"Processing {total_texts} texts on single GPU with batch size {batch_size}"
        )

        # Load model once
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(model_name, device=device)

        all_embeddings = []
        temp_files = []

        try:
            for i in range(0, total_texts, chunk_size):
                chunk_end = min(i + chunk_size, total_texts)
                chunk_texts = texts[i:chunk_end]
                print(f"Processing chunk {i//chunk_size + 1}: texts {i}-{chunk_end}")

                chunk_embeddings = []
                for j in range(0, len(chunk_texts), batch_size):
                    batch_end = min(j + batch_size, len(chunk_texts))
                    batch = chunk_texts[j:batch_end]

                    if is_query:
                        batch_embeddings = model.encode(
                            batch,
                            prompt_name="query",
                            batch_size=batch_size,
                            show_progress_bar=False,
                        )
                    else:
                        batch_embeddings = model.encode(
                            batch, batch_size=batch_size, show_progress_bar=False
                        )

                    chunk_embeddings.extend(batch_embeddings)

                    if j % (batch_size * 10) == 0:
                        print(f"  Chunk progress: {j}/{len(chunk_texts)}")
                        gc.collect()
                        torch.cuda.empty_cache()

                # Save chunk and clear memory
                temp_file = f"temp_chunk_{len(temp_files)}.npy"
                np.save(temp_file, np.array(chunk_embeddings))
                temp_files.append(temp_file)
                print(f"Saved chunk to {temp_file}")

                # Clear chunk from memory
                del chunk_embeddings
                gc.collect()
                torch.cuda.empty_cache()
                print_memory_usage(f"After chunk {i//chunk_size + 1}")

            # Combine all chunks
            print("Combining chunks...")
            for temp_file in temp_files:
                chunk = np.load(temp_file)
                all_embeddings.extend(chunk)
                os.remove(temp_file)

        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()

        return np.array(all_embeddings)

    def save_embeddings(embeddings, filename):
        """Save embeddings with memory cleanup"""
        np.save(filename, embeddings)
        print(f"Saved {len(embeddings)} embeddings to {filename}")
        # Clear the embeddings from memory after saving
        del embeddings
        gc.collect()

    # ============================================================================
    # CHEMHOTPOTQA RETRIEVAL EMBEDDINGS
    # ============================================================================

    # Generate and save corpus embeddings if not already present
    corpus_embeddings_path = "corpus_embeddings.npy"
    if not os.path.exists(corpus_embeddings_path):
        print("Generating corpus embeddings...")
        corpus_texts = ds_corpus["corpus"]["text"]
        print(f"Number of corpus texts: {len(corpus_texts)}")

        # Choose processing method based on available GPUs
        if torch.cuda.device_count() > 1:
            corpus_embeddings = get_embeddings_multi_gpu_sequential(
                corpus_texts,
                is_query=False,
                batch_size=2,  # Very small batch size for safety
            )
        else:
            corpus_embeddings = get_embeddings_single_gpu_chunked(
                corpus_texts, is_query=False, batch_size=4, chunk_size=500
            )

        save_embeddings(corpus_embeddings, corpus_embeddings_path)
        print(f"Corpus embeddings saved to {corpus_embeddings_path}")
        del corpus_embeddings  # Free memory
        gc.collect()
    else:
        print(f"{corpus_embeddings_path} already exists. Skipping generation.")

    print_memory_usage("After corpus embeddings")

    # Generate and save queries embeddings if not already present
    queries_embeddings_path = "queries_embeddings.npy"
    if not os.path.exists(queries_embeddings_path):
        print("Generating query embeddings...")
        query_texts = ds_queries["queries"]["text"]
        print(f"Number of query texts: {len(query_texts)}")

        # Choose processing method based on available GPUs
        if torch.cuda.device_count() > 1:
            queries_embeddings = get_embeddings_multi_gpu_sequential(
                query_texts,
                is_query=True,
                batch_size=2,  # Very small batch size for safety
            )
        else:
            queries_embeddings = get_embeddings_single_gpu_chunked(
                query_texts, is_query=True, batch_size=4, chunk_size=500
            )

        save_embeddings(queries_embeddings, queries_embeddings_path)
        print(f"Query embeddings saved to {queries_embeddings_path}")
        del queries_embeddings  # Free memory
        gc.collect()
    else:
        print(f"{queries_embeddings_path} already exists. Skipping generation.")

    print_memory_usage("After query embeddings")

    # Load embeddings from file for ChemHotpotQA evaluation
    print("Loading ChemHotpotQA embeddings from files...")
    corpus_embeddings = np.load(
        corpus_embeddings_path, mmap_mode="r"
    )  # Memory-mapped loading
    queries_embeddings = np.load(queries_embeddings_path, mmap_mode="r")

    print(f"Loaded corpus embeddings shape: {corpus_embeddings.shape}")
    print(f"Loaded query embeddings shape: {queries_embeddings.shape}")

    print_memory_usage("After loading ChemHotpotQA embeddings")

    # ============================================================================
    # CHEMHOTPOTQA EVALUATION
    # ============================================================================

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

    def compute_ndcg_fully_vectorized_memory_efficient(
        queries_embeddings, corpus_embeddings, ds_core, ds_corpus, k=10
    ):
        """
        Memory-efficient fully vectorized computation.
        """
        print("Starting ChemHotpotQA vectorized computation...")
        train_df = ds_core["train"].to_pandas()
        corpus_df = ds_corpus["corpus"].to_pandas()
        corpus_ids = np.array(corpus_df["_id"].tolist())

        num_queries_embeddings = len(queries_embeddings)
        num_queries_df = len(train_df)
        num_queries = min(num_queries_embeddings, num_queries_df)
        print(
            f"Processing {num_queries} queries (embeddings: {num_queries_embeddings}, df: {num_queries_df})"
        )

        # Process in chunks to avoid memory issues
        chunk_size = 50  # Process 50 queries at a time
        results = []

        for start_idx in range(0, num_queries, chunk_size):
            end_idx = min(start_idx + chunk_size, num_queries)
            print(f"Processing queries {start_idx} to {end_idx}")

            # Get chunk of queries
            query_chunk = queries_embeddings[start_idx:end_idx]
            df_chunk = train_df.iloc[start_idx:end_idx]

            # Compute similarities for chunk
            similarities_matrix = cosine_similarity(query_chunk, corpus_embeddings)
            top_k_indices = np.argsort(similarities_matrix, axis=1)[:, ::-1][:, :k]

            # Get ground truths for chunk
            query_ground_truths = df_chunk["corpus-id"].values
            top_k_corpus_ids = corpus_ids[top_k_indices]
            relevances_matrix = (
                top_k_corpus_ids == query_ground_truths[:, np.newaxis]
            ).astype(float)

            # Compute NDCG for chunk
            for i, (_, row) in enumerate(df_chunk.iterrows()):
                relevances = relevances_matrix[i]
                ndcg_score = compute_ndcg_at_k(relevances.tolist(), k)
                results.append(
                    {
                        "query-id": row["query-id"],
                        "ndcg_at_10": ndcg_score,
                    }
                )

            # Clean up
            del similarities_matrix, top_k_indices, relevances_matrix
            gc.collect()

            if start_idx % (chunk_size * 5) == 0:  # Less frequent memory reporting
                print_memory_usage(f"After chunk {start_idx}-{end_idx}")

        return results

    # Run ChemHotpotQA evaluation
    print("Starting ChemHotpotQA evaluation...")
    results_frame = pd.DataFrame(
        compute_ndcg_fully_vectorized_memory_efficient(
            queries_embeddings, corpus_embeddings, ds_core, ds_corpus
        )
    )

    mean_ndcg_10 = results_frame["ndcg_at_10"].mean()
    logger.debug(f"CHEMHOTPOTQA RETRIEVAL RESULTS: Mean NDCG@10: {mean_ndcg_10}")
    print(f"ChemHotpotQA Mean NDCG@10: {mean_ndcg_10}")

    # Clear ChemHotpotQA embeddings from memory
    del corpus_embeddings, queries_embeddings
    gc.collect()

    print_memory_usage("After ChemHotpotQA evaluation")

    # ============================================================================
    # CHEMEMB CLASSIFICATION
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
        print("Generating ChemEmb text embeddings...")
        print(f"Number of ChemEmb texts: {len(chememb_texts)}")

        # Choose processing method based on available GPUs
        if torch.cuda.device_count() > 1:
            chememb_text_embeddings = get_embeddings_multi_gpu_sequential(
                chememb_texts, is_query=False, batch_size=2
            )
        else:
            chememb_text_embeddings = get_embeddings_single_gpu_chunked(
                chememb_texts, is_query=False, batch_size=4, chunk_size=500
            )

        save_embeddings(chememb_text_embeddings, chememb_text_embeddings_path)
    else:
        logger.info(
            f"{chememb_text_embeddings_path} already exists. Skipping generation."
        )

    # Generate and save ChemEmb query embedding if not already present
    classification_query = "Is this text about chemistry? Yes or No?"
    if not os.path.exists(chememb_query_embedding_path):
        print("Generating ChemEmb query embedding...")

        # Load model for single query
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(model_name, device=device)

        try:
            query_embedding = model.encode(
                [classification_query], show_progress_bar=False
            )
            save_embeddings(query_embedding, chememb_query_embedding_path)
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
    else:
        logger.info(
            f"{chememb_query_embedding_path} already exists. Skipping generation."
        )

    print_memory_usage("After ChemEmb embeddings generation")

    # Load ChemEmb embeddings from file
    chememb_text_embeddings = np.load(chememb_text_embeddings_path, mmap_mode="r")
    chememb_query_embedding = np.load(chememb_query_embedding_path)[0]

    logger.debug(
        f"Processing {len(chememb_texts)} texts for chemistry classification..."
    )

    def classify_chemistry_content_vectorized(
        query_embedding, text_embeddings, threshold=0.0
    ):
        """
        Classify texts as chemistry-related based on cosine similarity with query.
        Higher similarity means more likely to be about chemistry.

        Args:
            query_embedding: Embedding of the classification query
            text_embeddings: Embeddings of texts to classify
            threshold: Similarity threshold for classification (default: 0.0)

        Returns:
            predictions: Binary predictions (1 for chemistry, 0 for not chemistry)
            similarities: Cosine similarity scores
        """
        # Compute similarities between query and all texts in chunks to avoid memory issues
        total_texts = len(text_embeddings)
        chunk_size = 1000
        all_similarities = []

        print(
            f"Computing similarities for {total_texts} texts in chunks of {chunk_size}"
        )

        for i in range(0, total_texts, chunk_size):
            end_idx = min(i + chunk_size, total_texts)
            chunk_embeddings = text_embeddings[i:end_idx]

            # Compute similarities for chunk
            chunk_similarities = cosine_similarity([query_embedding], chunk_embeddings)[
                0
            ]
            all_similarities.extend(chunk_similarities)

            if i % (chunk_size * 5) == 0:
                print(f"  Processed {i}/{total_texts} texts")

            # Clean up
            del chunk_embeddings, chunk_similarities
            gc.collect()

        similarities = np.array(all_similarities)

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

    # Write results to file
    output_filename = "model_scores.json"
    scores = {
        "model": model_name,
        "mean_ndcg_10": round(mean_ndcg_10, 4),
        "chememb_classification_accuracy": round(accuracy, 4),
        "chememb_precision": round(precision, 4),
        "chememb_recall": round(recall, 4),
        "chememb_f1": round(f1, 4),
    }

    with open(output_filename, "w") as f:
        json.dump(scores, f, indent=2)

    logger.debug(f"Results written to {output_filename}")
    print(
        f"Final Results - ChemHotpotQA NDCG@10: {mean_ndcg_10:.4f}, ChemEmb Accuracy: {accuracy:.4f}"
    )

    print_memory_usage("Final")


if __name__ == "__main__":
    main()

# This script contains functions for loading and preprocessing the dataset.
import pandas as pd
import numpy as np
import os
from typing import List
import re
import string
from transformers import AutoTokenizer
import torch
import numpy as np
from datasets import Dataset
import config


def load_specific_language_data(data_dir: str, lang_codes: List[str]) -> pd.DataFrame:
    """
    Loads and combines data for specific languages from the BRIGHTER dataset.

    Args:
        data_dir (str): The directory containing the language CSV files.
        lang_codes (List[str]): A list of language codes to load (e.g., ['amh', 'hau']).

    Returns:
        pd.DataFrame: A single DataFrame containing the combined data.
    """
    all_dfs = []
    for lang in lang_codes:
        file_path = os.path.join(data_dir, f"{lang}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df["language"] = lang
            all_dfs.append(df)
            print(f"Loaded {lang} data with {len(df)} samples.")
        else:
            print(f"Warning: Data file not found for language '{lang}' at {file_path}")

    if not all_dfs:
        print("No data loaded. Returning empty DataFrame.")
        return pd.DataFrame()

    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Total combined samples: {len(combined_df)}")
    return combined_df


def analyze_and_clean_data(df, label_columns):
    """
    Analyze data quality and remove potentially noisy samples.
    """
    print("Analyzing data quality...")

    # this Checks emotions distribution
    emotion_counts = df[label_columns].sum(axis=1)
    print(f"Samples with 0 emotions: {(emotion_counts == 0).sum()}")
    print(f"Samples with 1 emotion: {(emotion_counts == 1).sum()}")
    print(f"Samples with 2 emotions: {(emotion_counts == 2).sum()}")
    print(f"Samples with 3+ emotions: {(emotion_counts >= 3).sum()}")

    # Removessamples with too many emotions (likely noisy)
    clean_df = df[emotion_counts <= 2].copy()  # Keep max 2 emotions
    print(f"Removed {len(df) - len(clean_df)} potentially noisy samples")

    # Checking for very short texts (likely not informative)
    short_texts = clean_df["text"].str.len() < 10
    clean_df = clean_df[~short_texts]
    print(f"Removed {short_texts.sum()} very short texts")

    return clean_df


def preprocess_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies preprocessing steps to the text column.

    Args:
        df (pd.DataFrame): The input DataFrame with a 'text' column.

    Returns:
        pd.DataFrame: The DataFrame with a new 'processed_text' column.
    """
    print("Applying text preprocessing...")

    # to ensure 'text' column is string type, handling potential float/NaN values
    df["processed_text"] = df["text"].astype(str)

    # 1. lowercase the text
    df["processed_text"] = df["processed_text"].str.lower()

    # 2.remove URLs
    df["processed_text"] = df["processed_text"].apply(
        lambda x: re.sub(r"http\S+|www\S+", "", x)
    )

    # 3.remove user mentions (@)
    df["processed_text"] = df["processed_text"].apply(lambda x: re.sub(r"@\w+", "", x))

    # 4. Refined punctuation removal.
    punct_to_remove = f"[{re.escape(string.punctuation)}]"
    df["processed_text"] = df["processed_text"].apply(
        lambda x: re.sub(punct_to_remove, "", x)
    )

    # 5. Normalize whitespace.
    df["processed_text"] = df["processed_text"].apply(
        lambda x: re.sub(r"\s+", " ", x).strip()
    )

    print("Text preprocessing complete.")
    return df


def create_dataset(df: pd.DataFrame, model_name: str):
    """
    Cleans, tokenizes, and prepares data for multi-label classification.
    Ensures all emotion labels from config.LABEL_COLUMNS are included in the output,
    even if they don't exist in the input data (will be filled with zeros).

    Args:
        df (pd.DataFrame): The input DataFrame with a 'processed_text' column.
        model_name (str): The identifier for the pre-trained model's tokenizer.

    Returns:
        A tuple containing the training dataset, evaluation dataset, and label columns.
    """
    # 1. Load Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 2. Clean data using the globally defined function
    emotion_columns = config.LABEL_COLUMNS

    # Ensure all required emotion columns exist in the DataFrame
    for col in emotion_columns:
        if col not in df.columns:
            print(f"Adding missing emotion column: {col} (filled with 0s)")
            df[col] = 0

    # Ensure we only keep the columns we want and in the correct order
    df = df[["text", "processed_text"] + emotion_columns].copy()

    df_clean = analyze_and_clean_data(df, emotion_columns)

    # 3. Prepare Labels - ensure all emotion columns are present and in correct order
    labels = df_clean[emotion_columns].values.astype(float).tolist()
    df_clean["labels"] = labels
    print(
        f"Labels prepared for multi-label classification. Using {len(emotion_columns)} emotion categories."
    )

    # 4. Create Hugging Face Dataset
    dataset = Dataset.from_pandas(df_clean[["processed_text", "labels"]])

    # 5. Tokenize Text
    print("Tokenizing text...")

    def tokenize_function(examples):
        return tokenizer(
            examples["processed_text"],
            padding="max_length",
            truncation=True,
            max_length=config.MAX_LENGTH,
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["processed_text"])
    tokenized_dataset.set_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )

    # 6. Split Dataset
    split_ratio = 1.0 - config.TRAIN_TEST_SPLIT_RATIO
    train_test_split = tokenized_dataset.train_test_split(
        test_size=split_ratio, seed=config.SEED
    )
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]
    print(
        f"Dataset split into training ({len(train_dataset)} samples) and validation ({len(eval_dataset)} samples) sets."
    )

    return train_dataset, eval_dataset, emotion_columns


def calculate_class_weights(df, label_columns):
    """
    Calculates class weights for handling class imbalance in multi-label classification.
    The weight for a class is the ratio of negative to positive instances.
    This is used as the `pos_weight` argument in BCEWithLogitsLoss.

    Args:
        df (pd.DataFrame): The dataframe containing the training data.
        label_columns (list): A list of strings with the names of the label columns.

    Returns:
        torch.Tensor: A tensor containing the calculated weight for each class.
    """
    print("Calculating class weights for imbalance...")
    num_samples = len(df)
    pos_counts = df[label_columns].sum()
    neg_counts = num_samples - pos_counts
    pos_weights = neg_counts / pos_counts

    # Replace inf with a large number if a class has zero positive instances, though this shouldn't happen with good data.
    pos_weights = pos_weights.replace([np.inf, -np.inf], 0).fillna(0)

    weights = torch.tensor(pos_weights.values, dtype=torch.float)
    print("Calculated weights:", weights)
    return weights

# This script contains functions for loading and preprocessing the dataset.
import pandas as pd
import os
from typing import List
import re
import string
from transformers import AutoTokenizer
from datasets import Dataset

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


def preprocess_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies preprocessing steps to the text column.

    Args:
        df (pd.DataFrame): The input DataFrame with a 'text' column.

    Returns:
        pd.DataFrame: The DataFrame with a new 'processed_text' column.
    """
    print("Applying text preprocessing...")

    # Ensure 'text' column is string type, handling potential float/NaN values
    df['processed_text'] = df['text'].astype(str)

    # 1. Lowercase the text
    df['processed_text'] = df['processed_text'].str.lower()

    # 2. Remove URLs
    df['processed_text'] = df['processed_text'].apply(lambda x: re.sub(r'http\S+|www\S+', '', x))

    # 3. Remove user mentions (@)
    df['processed_text'] = df['processed_text'].apply(lambda x: re.sub(r'@\w+', '', x))

    # 4. Refined punctuation removal.
    # This regex removes characters from the standard `string.punctuation` set,
    # which is suitable for cleaning text while preserving language-specific characters.
    punct_to_remove = f'[{re.escape(string.punctuation)}]' 
    df['processed_text'] = df['processed_text'].apply(lambda x: re.sub(punct_to_remove, '', x))

    # 5. Normalize whitespace.
    # This replaces multiple whitespace characters with a single space and removes leading/trailing spaces.
    df['processed_text'] = df['processed_text'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())

    print("Text preprocessing complete.")
    return df

def create_dataset(df: pd.DataFrame, model_name: str):
    """
    Tokenizes text and prepares labels for multi-label classification,
    then creates and splits a Hugging Face Dataset.

    Args:
        df (pd.DataFrame): The preprocessed DataFrame.
        model_name (str): The identifier for the pre-trained model's tokenizer.

    Returns:
        A tuple containing the training and validation Datasets.
    """
    # 1. Load Tokenizer
    # The tokenizer is responsible for converting raw text into a format the model can understand.
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 2. Prepare Labels
    # For multi-label classification, the model expects a list of floats for each text entry.
    emotion_columns = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
    labels = df[emotion_columns].values.astype(float).tolist()
    df['labels'] = labels
    print("Labels prepared for multi-label classification.")

    # 3. Create Hugging Face Dataset from DataFrame
    # We convert the pandas DataFrame to a Hugging Face Dataset object.
    dataset = Dataset.from_pandas(df[['processed_text', 'labels']])
    
    # 4. Tokenize Text
    # We apply the tokenizer to all texts in the 'processed_text' column.
    # Padding ensures all sequences have the same length, and truncation cuts longer sequences.
    print("Tokenizing text...")
    def tokenize_function(examples):
        return tokenizer(examples['processed_text'], padding='max_length', truncation=True, max_length=128)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Remove the original text column as it's no longer needed after tokenization.
    tokenized_dataset = tokenized_dataset.remove_columns(['processed_text'])
    
    # Set the format to 'torch' to get PyTorch tensors.
    tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    print("Hugging Face Dataset created.")

    # 5. Split Dataset
    # We split the dataset into training and validation sets to evaluate the model's performance.
    train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']
    print("Dataset split into training and validation sets.")

    return train_dataset, eval_dataset

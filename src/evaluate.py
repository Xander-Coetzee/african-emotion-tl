# For calculating and reporting performance metrics.
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (precision_recall_fscore_support, 
                             multilabel_confusion_matrix, hamming_loss, accuracy_score)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

import config
from src.data_preprocessing import create_dataset, load_specific_language_data, preprocess_text

def compute_metrics(eval_pred):
    """
    Computes and returns a dictionary of metrics for evaluation.

    Args:
        eval_pred (EvalPrediction): A tuple containing the model's predictions and the true labels.

    Returns:
        dict: A dictionary of performance metrics.
    """
    # Extract logits and labels from the EvalPrediction object
    logits, labels = eval_pred

    # Apply sigmoid to convert logits to probabilities
    probs = 1 / (1 + np.exp(-logits))

    # Use a 0.5 threshold to get binary predictions
    binary_preds = (probs > 0.5).astype(int)

    # --- Calculate Overall Metrics ---
    # Calculate precision, recall, and F1-score with different averaging methods
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(labels, binary_preds, average='micro')
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(labels, binary_preds, average='macro', zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(labels, binary_preds, average='weighted', zero_division=0)
    
    # Calculate Hamming loss and subset accuracy
    h_loss = hamming_loss(labels, binary_preds)
    acc = accuracy_score(labels, binary_preds)

    # --- Compile Metrics into a Dictionary ---
    metrics = {
        'accuracy_subset': acc,
        'hamming_loss': h_loss,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_micro': p_micro,
        'precision_macro': p_macro,
        'precision_weighted': p_weighted,
        'recall_micro': r_micro,
        'recall_macro': r_macro,
        'recall_weighted': r_weighted,
    }
    
    return metrics

def main():
    """Main function to run the detailed evaluation."""
    print("Starting detailed model evaluation...")

    # --- 1. Load Tokenizer and Data First ---
    base_model_name = config.MODELS["afrixlmr"]
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Load and preprocess the data in the same way as in run.py
    print("Loading and preprocessing data...")
    raw_data_path = os.path.join(config.RAW_DATA_PATH, "SemEval2025-Task11", "task-dataset", "semeval-2025-task11-dataset", "track_a", "train")
    df = load_specific_language_data(raw_data_path, config.LANGUAGES)
    df = preprocess_text(df)

    _, eval_dataset, label_columns = create_dataset(df, tokenizer)
    num_labels = len(label_columns)
    print(f"Loaded and processed validation data. Found {num_labels} labels: {label_columns}")

    # --- 2. Load Model and Adapter ---
    model_path = "./results/checkpoint-1080"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the base model with the correct number of labels
    model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=num_labels)
    
    # Load the adapter from the checkpoint
    model.load_adapter(model_path, "emotion_classification")
    model.set_active_adapters("emotion_classification")
    model.to(device)
    print(f"Loaded base model '{base_model_name}' and attached adapter from: {model_path}")

    # --- 3. Get Predictions ---
    trainer = Trainer(model=model)
    raw_predictions = trainer.predict(eval_dataset)
    
    # Apply sigmoid and threshold to get binary predictions
    logits = raw_predictions.predictions[0]
    labels = raw_predictions.label_ids
    probs = 1 / (1 + np.exp(-logits))
    binary_preds = (probs > 0.5).astype(int)

    # --- 4. Overall Metrics ---
    print("\n--- Overall Performance Metrics ---")
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(labels, binary_preds, average='micro')
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(labels, binary_preds, average='macro', zero_division=0)
    h_loss = hamming_loss(labels, binary_preds)
    acc = accuracy_score(labels, binary_preds)

    print(f"Accuracy (Subset): {acc:.4f}")
    print(f"Hamming Loss: {h_loss:.4f}")
    print(f"F1 Score (Micro): {f1_micro:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"Precision (Micro): {p_micro:.4f}")
    print(f"Recall (Micro): {r_micro:.4f}")

    # --- 5. Per-Class Metrics ---
    print("\n--- Per-Class Performance ---")
    p, r, f1, s = precision_recall_fscore_support(labels, binary_preds, average=None, labels=list(range(len(label_columns))), zero_division=0)
    
    # Create a DataFrame for pretty printing
    metrics_df = pd.DataFrame({
        'Emotion': label_columns,
        'Precision': p,
        'Recall': r,
        'F1-Score': f1,
        'Support': s
    })
    print(metrics_df.to_string(index=False))

    # --- 6. Confusion Matrices ---
    print("\n--- Per-Class Confusion Matrices ---")
    mcm = multilabel_confusion_matrix(labels, binary_preds)
    for i, label in enumerate(label_columns):
        print(f"\nConfusion Matrix for: '{label}'")
        tn, fp, fn, tp = mcm[i].ravel()
        cm_df = pd.DataFrame([
            [f"TN = {tn}", f"FP = {fp}"],
            [f"FN = {fn}", f"TP = {tp}"]
        ], index=["Actual Negative", "Actual Positive"], columns=["Predicted Negative", "Predicted Positive"])
        print(cm_df)

if __name__ == "__main__":
    main()

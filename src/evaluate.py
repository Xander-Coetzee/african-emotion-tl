# For calculating and reporting performance metrics.
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support,
    multilabel_confusion_matrix,
    hamming_loss,
    accuracy_score,
    precision_recall_fscore_support,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

import config
from src.data_preprocessing import (
    create_dataset,
    load_specific_language_data,
    preprocess_text,
)


def compute_metrics(eval_pred):
    """
    Computes and returns a dictionary of metrics for evaluation.

    Args:
        eval_pred (EvalPrediction): A tuple containing the model's predictions and the true labels.

    Returns:
        dict: A dictionary of performance metrics, including per-emotion scores.
    """
    # extract logits and labels from the EvalPrediction object
    logits, labels = eval_pred

    # apply sigmoid to convert logits to probabilities
    probs = 1 / (1 + np.exp(-logits))

    # Use a 0.5 thresholdd to get binary predictions
    binary_preds = (probs > 0.5).astype(int)

    # Calculate Overall Metrics
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        labels, binary_preds, average="micro", zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, binary_preds, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, binary_preds, average="weighted", zero_division=0
    )
    h_loss = hamming_loss(labels, binary_preds)
    acc = accuracy_score(labels, binary_preds)

    # Compile Overall Metrics
    metrics = {
        "accuracy_subset": acc,
        "hamming_loss": h_loss,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_micro": p_micro,
        "precision_macro": p_macro,
        "precision_weighted": p_weighted,
        "recall_micro": r_micro,
        "recall_macro": r_macro,
        "recall_weighted": r_weighted,
    }

    # Calculate and Add Per-Emotion Metrics
    p_class, r_class, f1_class, s_class = precision_recall_fscore_support(
        labels, binary_preds, zero_division=0
    )

    # Add per-emotion metrics to the dictionary
    for i, label in enumerate(config.LABEL_COLUMNS):
        metrics[f"precision_{label}"] = p_class[i]
        metrics[f"recall_{label}"] = r_class[i]
        metrics[f"f1_{label}"] = f1_class[i]
        metrics[f"support_{label}"] = s_class[i]

    return metrics

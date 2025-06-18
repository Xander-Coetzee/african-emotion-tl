#!/usr/bin/env python3
"""
Main script for running emotion classification experiments.

This script handles data loading, model training, evaluation, and saving results.
It supports multiple models and languages through command-line arguments.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    hamming_loss,
)
from transformers import Trainer, TrainingArguments

# Local imports
import config
from src import data_preprocessing, model, train, evaluate, utils

# Set up logging
print("--- SCRIPT EXECUTION STARTED ---", file=sys.stderr)


def parse_arguments() -> argparse.Namespace:
    """Parse and return command line arguments.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Train and evaluate emotion classification models.'
    )
    
    # Model and language configuration
    parser.add_argument(
        '--model', 
        type=str, 
        default=config.MODEL_NAME,
        help=f'Model name or path (default: {config.MODEL_NAME})'
    )
    parser.add_argument(
        '--lang', 
        type=str, 
        default=config.LANG_CODE,
        choices=['hau', 'eng'],
        help=f'Language code (default: {config.LANG_CODE})'
    )
    
    # Experiment configuration
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='results',
        help='Directory to save results (default: results/)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=config.TRAINING_ARGS.get('per_device_train_batch_size', 8),
        help=f'Batch size for training and evaluation (default: {config.TRAINING_ARGS.get("per_device_train_batch_size", 8)})'
    )
    
    # Training control
    parser.add_argument(
        '--epochs',
        type=int,
        default=config.TRAINING_ARGS.get('num_train_epochs', 3),
        help=f'Number of training epochs (default: {config.TRAINING_ARGS.get("num_train_epochs", 3)})'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=config.TRAINING_ARGS.get('learning_rate', 2e-5),
        help=f'Learning rate (default: {config.TRAINING_ARGS.get("learning_rate", 2e-5)})'
    )
    
    # Additional options
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Run error analysis instead of training'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    return parser.parse_args()


def save_results(
    metrics: Dict[str, Any], 
    output_dir: str, 
    model_name: str, 
    lang: str,
    **additional_metadata: Dict[str, Any]
) -> str:
    """Save evaluation metrics and metadata to a JSON file.
    
    Args:
        metrics: Dictionary of evaluation metrics
        output_dir: Directory to save results
        model_name: Name of the model used
        lang: Language code
        **additional_metadata: Additional metadata to include
        
    Returns:
        Path to the saved results file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean up model name for filename
    model_name_clean = model_name.replace('/', '_').replace('\\', '_')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create filename
    filename = os.path.join(output_dir, f"{model_name_clean}_{lang}_{timestamp}.json")
    
    # Prepare result data with metadata
    result_data = {
        'experiment_metadata': {
            'model': model_name,
            'language': lang,
            'timestamp': datetime.now().isoformat(),
            'git_commit': utils.get_git_commit_hash(),
            'command': ' '.join(sys.argv),
            **additional_metadata
        },
        'metrics': metrics
    }
    
    # Save to file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to: {filename}")
    return filename

# This is the main execution script for the data processing pipeline.
# It orchestrates the loading, preprocessing, and saving of the dataset.


def apply_precision_rules(predictions, probabilities, label_columns):
    """
    Apply rule-based post-processing to improve precision.

    Args:
        predictions: Binary predictions array (n_samples, n_labels)
        probabilities: Probability scores array (n_samples, n_labels)
        label_columns: List of emotion labels

    Returns:
        Modified predictions array
    """
    corrected_preds = predictions.copy()
    emotion_to_idx = {emotion: i for i, emotion in enumerate(label_columns)}

    print("Applying precision improvement rules...")

    # Rule 1: Limit maximum emotions per sample (reduces false positives)
    max_emotions = 2  # Most samples should have 1-2 emotions
    samples_modified = 0

    for i in range(len(predictions)):
        active_emotions = predictions[i].sum()

        if active_emotions > max_emotions:
            # Keep only the top emotions based on probability
            emotion_probs = probabilities[i]
            top_indices = np.argsort(emotion_probs)[-max_emotions:]

            # Reset all predictions for this sample
            corrected_preds[i] = 0
            # Set only top emotions
            corrected_preds[i, top_indices] = 1
            samples_modified += 1

    print(f"  Rule 1: Limited emotions for {samples_modified} samples")

    # Rule 2: Handle contradictory emotions
    contradictory_pairs = [("joy", "sadness"), ("joy", "anger"), ("joy", "disgust")]

    conflicts_resolved = 0
    for joy_emotion, negative_emotion in contradictory_pairs:
        if joy_emotion in emotion_to_idx and negative_emotion in emotion_to_idx:
            joy_idx = emotion_to_idx[joy_emotion]
            neg_idx = emotion_to_idx[negative_emotion]

            # Find samples with both emotions predicted
            both_predicted = (corrected_preds[:, joy_idx] == 1) & (
                corrected_preds[:, neg_idx] == 1
            )

            for i in np.where(both_predicted)[0]:
                # Keep the emotion with higher probability
                if probabilities[i, joy_idx] > probabilities[i, neg_idx]:
                    corrected_preds[i, neg_idx] = 0
                else:
                    corrected_preds[i, joy_idx] = 0
                conflicts_resolved += 1

    print(f"  Rule 2: Resolved {conflicts_resolved} contradictory emotion pairs")

    # Rule 3: Confidence-based filtering
    # Remove predictions with very low confidence
    min_confidence = 0.6
    low_confidence_removed = 0

    for i in range(len(predictions)):
        for j in range(len(label_columns)):
            if corrected_preds[i, j] == 1 and probabilities[i, j] < min_confidence:
                corrected_preds[i, j] = 0
                low_confidence_removed += 1

    print(f"  Rule 3: Removed {low_confidence_removed} low-confidence predictions")

    # Rule 4: Ensure at least one emotion (if original had emotions)
    samples_with_no_emotions = 0
    for i in range(len(predictions)):
        if predictions[i].sum() > 0 and corrected_preds[i].sum() == 0:
            # Restore the highest probability emotion
            best_emotion_idx = np.argmax(probabilities[i])
            corrected_preds[i, best_emotion_idx] = 1
            samples_with_no_emotions += 1

    print(f"  Rule 4: Restored emotions for {samples_with_no_emotions} samples")

    return corrected_preds


def run_analysis():
    """Main function to run the error analysis pipeline with precision improvements."""
    print("--- Starting Error Analysis Pipeline ---")

    # --- Data Loading ---
    processed_file_path = os.path.join(config.PROCESSED_DATA_PATH, "hau_processed.csv")
    if not os.path.exists(processed_file_path):
        print(f"Error: Preprocessed data not found at {processed_file_path}.")
        print("Please run the training pipeline first to generate the data.")
        return

    print(f"Loading preprocessed data from: {processed_file_path}")
    df = pd.read_csv(processed_file_path)

    # --- Data Splitting ---
    utils.set_seed(config.SEED)
    train_df = df.sample(frac=config.TRAIN_TEST_SPLIT_RATIO, random_state=config.SEED)
    val_df = df.drop(train_df.index)
    print(f"Validation set size: {len(val_df)}")

    # --- Model and Tokenizer Loading ---
    print(f"Loading model: {config.MODEL_NAME}", flush=True)
    analysis_model = model.AutoAdapterModel.from_pretrained(config.MODEL_NAME)
    print("DEBUG: Base model loaded.", flush=True)

    adapter_path = os.path.join(config.MODEL_OUTPUT_DIR, config.ADAPTER_NAME)
    if not os.path.exists(adapter_path):
        print(f"Error: Trained adapter not found at {adapter_path}")
        print("Please run the training pipeline first.")
        return

    analysis_model.load_adapter(adapter_path, with_head=True)
    print("DEBUG: Adapter loaded.", flush=True)
    analysis_model.set_active_adapters(config.ADAPTER_NAME)
    print("DEBUG: Adapter set to active.", flush=True)
    print(f"Loaded adapter '{config.ADAPTER_NAME}' from {adapter_path}")

    print("Loading tokenizer...", flush=True)
    tokenizer = data_preprocessing.AutoTokenizer.from_pretrained(config.MODEL_NAME)
    print("DEBUG: Tokenizer loaded.", flush=True)

    # --- Dataset Preparation ---
    labels = val_df[config.LABEL_COLUMNS].values.astype(float).tolist()
    val_df["labels"] = labels

    val_dataset = Dataset.from_pandas(val_df[["processed_text", "labels", "text"]])

    def tokenize_function(examples):
        return tokenizer(
            examples["processed_text"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    print("Tokenizing validation set...")
    print("DEBUG: Starting tokenization map function.", flush=True)
    tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)
    print("DEBUG: Tokenization finished.", flush=True)
    tokenized_val_dataset = tokenized_val_dataset.remove_columns(["processed_text"])
    tokenized_val_dataset.set_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )

    # --- Prediction and Analysis ---
    print("DEBUG: Initializing Trainer.", flush=True)
    trainer = Trainer(model=analysis_model)
    print("DEBUG: Trainer initialized.", flush=True)

    print("Running predictions...", flush=True)
    preds = trainer.predict(tokenized_val_dataset)
    print("DEBUG: Predictions finished.", flush=True)
    logits = preds.predictions
    true_labels = preds.label_ids

    probs = 1 / (1 + np.exp(-logits))

    # NEW: Per-class threshold optimization for precision
    print("Finding optimal per-class thresholds for precision...")
    optimal_thresholds = {}
    precision_results = {}

    for i, emotion in enumerate(config.LABEL_COLUMNS):
        best_score = 0
        best_threshold = 0.5
        best_metrics = {}

        print(f"\nOptimizing threshold for {emotion}...")

        for threshold in np.arange(
            0.3, 0.85, 0.01
        ):  # Higher threshold range for precision
            binary_preds = (probs[:, i] > threshold).astype(int)

            # Calculate metrics
            precision, recall, f1, support = precision_recall_fscore_support(
                true_labels[:, i], binary_preds, average="binary", zero_division=0
            )

            # Precision-weighted score (prioritize precision over recall)
            if precision > 0 and recall > 0:
                precision_weight = 0.7  # Give more weight to precision
                recall_weight = 0.3
                weighted_score = precision_weight * precision + recall_weight * recall
            else:
                weighted_score = 0

            if weighted_score > best_score:
                best_score = weighted_score
                best_threshold = threshold
                best_metrics = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "weighted_score": weighted_score,
                }

        optimal_thresholds[emotion] = best_threshold
        precision_results[emotion] = best_metrics

        print(
            f"  {emotion}: threshold={best_threshold:.3f}, "
            f"precision={best_metrics['precision']:.4f}, "
            f"recall={best_metrics['recall']:.4f}, "
            f"f1={best_metrics['f1']:.4f}"
        )

    # Apply optimal thresholds
    print("\nApplying optimal per-class thresholds...")
    binary_preds = np.zeros_like(probs)
    for i, emotion in enumerate(config.LABEL_COLUMNS):
        binary_preds[:, i] = (probs[:, i] > optimal_thresholds[emotion]).astype(int)

    # NEW: Apply post-processing rules to improve precision
    print("Applying post-processing rules...")
    binary_preds = apply_precision_rules(binary_preds, probs, config.LABEL_COLUMNS)

    print("\n--- Classification Report (With Precision Optimization) ---")
    report = classification_report(
        true_labels, binary_preds, target_names=config.LABEL_COLUMNS, zero_division=0
    )
    print(report)

    # Calculate comprehensive metrics
    subset_accuracy = (true_labels == binary_preds).all(axis=1).mean()
    hamming_loss_score = np.mean(true_labels != binary_preds)

    print(f"\n--- Precision-Optimized Metrics ---")
    print(f"Subset Accuracy: {subset_accuracy:.4f}")
    print(f"Hamming Loss: {hamming_loss_score:.4f}")

    # Per-emotion detailed metrics
    print(f"\n--- Per-Emotion Precision Analysis ---")
    for i, emotion in enumerate(config.LABEL_COLUMNS):
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels[:, i], binary_preds[:, i], average="binary", zero_division=0
        )
        print(
            f"{emotion:>10}: Precision={precision:.4f}, Recall={recall:.4f}, "
            f"F1={f1:.4f}, Threshold={optimal_thresholds[emotion]:.3f}"
        )

    # Save results with threshold information
    true_labels_df = pd.DataFrame(
        true_labels, columns=[f"{col}_true" for col in config.LABEL_COLUMNS]
    )
    pred_labels_df = pd.DataFrame(
        binary_preds, columns=[f"{col}_pred" for col in config.LABEL_COLUMNS]
    )
    probs_df = pd.DataFrame(
        probs, columns=[f"{col}_prob" for col in config.LABEL_COLUMNS]
    )

    val_df.reset_index(drop=True, inplace=True)
    true_labels_df.reset_index(drop=True, inplace=True)
    pred_labels_df.reset_index(drop=True, inplace=True)
    probs_df.reset_index(drop=True, inplace=True)

    review_df = pd.concat(
        [val_df[["text"]], true_labels_df, pred_labels_df, probs_df], axis=1
    )
    review_df["is_correct"] = (
        review_df[[f"{col}_true" for col in config.LABEL_COLUMNS]].values
        == review_df[[f"{col}_pred" for col in config.LABEL_COLUMNS]].values
    ).all(axis=1)

    # Add threshold information to results
    threshold_info = pd.DataFrame([optimal_thresholds]).add_suffix("_threshold")

    output_path = os.path.join(config.OUTPUT_DIR, "precision_optimized_results.csv")
    review_df.to_csv(output_path, index=False)

    # Save threshold information separately
    threshold_path = os.path.join(config.OUTPUT_DIR, "optimal_thresholds.csv")
    threshold_info.to_csv(threshold_path, index=False)

    print(f"\nSaved detailed predictions to: {output_path}")
    print(f"Saved optimal thresholds to: {threshold_path}")
    print(f"Number of incorrect predictions: {len(review_df[~review_df['is_correct']])}")
    return parser.parse_args()

def save_results(metrics, output_dir, model_name, lang):
    """Save evaluation metrics to a JSON file."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean up model name for filename
    model_name_clean = model_name.split('/')[-1]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create filename
    filename = f"{output_dir}/final_metrics_{model_name_clean}_{lang}_{timestamp}.json"
    
    # Add metadata
    metrics['experiment_metadata'] = {
        'model': model_name,
        'language': lang,
        'timestamp': datetime.now().isoformat(),
    }
    
    # Save to file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to: {filename}")
    return filename

def evaluate_model(trainer, eval_dataset, label_columns):
    """Evaluate the model and return metrics.
    
    Args:
        trainer: Initialized Trainer instance
        eval_dataset: Dataset for evaluation
        label_columns: List of label column names
        
    Returns:
        Dictionary of evaluation metrics
    """
    print("\n--- Evaluating model ---")
    
    # Get predictions and labels
    predictions = trainer.predict(eval_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids
    
    # Calculate metrics
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )
    
    # Calculate per-class metrics
    metrics = {
        'accuracy': accuracy,
        'precision_weighted': precision,
        'recall_weighted': recall,
        'f1_weighted': f1,
        'hamming_loss': hamming_loss(labels, preds),
    }
    
    # Add per-class metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )
    metrics.update({
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
    })
    
    # Print metrics
    print("\n--- Evaluation Results ---")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics

def main():
    """Main function to run the data processing pipeline."""
    print("=== Script started successfully ===")
    
    # Parse command line arguments
    print("\n--- Parsing command line arguments ---")
    args = parse_arguments()
    print(f"Arguments parsed: {args}")
    
    # Override config with command line arguments
    config.MODEL_NAME = args.model
    config.LANG_CODE = args.lang
    config.TRAINING_ARGS['per_device_train_batch_size'] = args.batch_size
    config.TRAINING_ARGS['per_device_eval_batch_size'] = args.batch_size
    config.TRAINING_ARGS['num_train_epochs'] = args.epochs
    config.TRAINING_ARGS['learning_rate'] = args.learning_rate
    
    # Set up logging
    print("\n=== Starting Experiment ===")
    print(f"Model: {config.MODEL_NAME}")
    print(f"Language: {config.LANG_CODE}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Output directory: {args.output_dir}")
    
    # Ensure output directories exist
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    utils.set_seed(args.seed)
    
    # Load and preprocess data
    print("\n--- Loading and preprocessing data ---")
    try:
        # Load the dataset
        df = data_preprocessing.load_specific_language_data(
            config.RAW_DATA_PATH, 
            [config.LANG_CODE]
        )
        
        if df.empty:
            print(f"No data found for language: {config.LANG_CODE}")
            return
            
        # Preprocess text
        df = data_preprocessing.preprocess_text(df)
        
        # Create dataset splits
        train_dataset, eval_dataset, label_columns = data_preprocessing.create_dataset(
            df, 
            config.MODEL_NAME
        )
        
        # Initialize model
        print("\n--- Initializing model ---")
        num_labels = len(label_columns)
        model_instance = model.get_model(config.MODEL_NAME, num_labels)
        
        # Train model
        print("\n--- Starting training ---")
        trainer = train.train_model(
            model_instance,
            train_dataset,
            eval_dataset,
            class_weights=None
        )
        
        # Evaluate model
        print("\n--- Evaluating model ---")
        metrics = evaluate_model(trainer, eval_dataset, label_columns)
        
        # Save results
        results_file = save_results(
            metrics=metrics,
            output_dir=args.output_dir,
            model_name=config.MODEL_NAME,
            lang=config.LANG_CODE,
            training_args={
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'learning_rate': args.learning_rate,
                'seed': args.seed
            }
        )
        print(f"Results saved to: {results_file}")
        
    except Exception as e:
        print(f"\n--- ERROR: An exception occurred during execution ---")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # If the --analyze flag is passed, run the error analysis and exit.
    if args.analyze:
        try:
            print("Attempting to run analysis...")
            run_analysis()
            print("Analysis function finished successfully.")
        except BaseException as e:
            import traceback
            print(f"--- ERROR: An exception occurred during analysis ---")
            traceback.print_exc()
        return

    # Print final metrics if they exist
    if 'final_metrics' in locals():
        print("\n--- Final Metrics ---")
        print(f"  Accuracy:           {final_metrics.get('eval_accuracy', 0):.4f}")
        print(f"  F1 (Weighted):      {final_metrics.get('eval_f1_weighted', 0):.4f}")
        print(f"  Precision (Weighted): {final_metrics.get('eval_precision_weighted', 0):.4f}")
        print(f"  Recall (Weighted):   {final_metrics.get('eval_recall_weighted', 0):.4f}")

        print("\n--- Per-Emotion Metrics ---")
        for label in config.LABEL_COLUMNS:
            print(f"\nEmotion: {label}")
            print(f"  Precision: {final_metrics.get(f'eval_precision_{label}', 0):.4f}")
            print(f"  Recall:    {final_metrics.get(f'eval_recall_{label}', 0):.4f}")
            print(f"  F1-Score:  {final_metrics.get(f'eval_f1_{label}', 0):.4f}")
            print(f"  Support:   {final_metrics.get(f'eval_support_{label}', 0)}")

    print("\nExperiment pipeline finished.")


if __name__ == "__main__":
    main()

# evaluate_model.py
# This script loads a trained adapter model from a checkpoint and evaluates it
# on the validation set, printing comprehensive multilabel classification metrics.

import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from tqdm import tqdm
import numpy as np

import config
from src import data_preprocessing # To access create_dataset for eval data
from src.evaluate import compute_metrics # Our metrics function
from src.model import get_model # To load the base model structure before loading adapter

def main():
    print("Starting model evaluation...")

    # --- Configuration ---
    model_name = config.MODELS['afrixlmr']
    checkpoint_path = os.path.join(".", "results", "checkpoint-324", "emotion_classification") # Correct path to the adapter itself
    num_labels = 6  # anger, disgust, fear, joy, sadness, surprise
    lang_codes_to_load = config.LANGUAGES
    raw_data_path = os.path.join(config.RAW_DATA_PATH, "SemEval2025-Task11", "task-dataset", "semeval-2025-task11-dataset", "track_a", "train")

    # --- Device Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # --- Load Tokenizer ---
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # --- Load and Prepare Evaluation Data ---
    print("Loading and preprocessing evaluation data...")
    # Load the full dataset first (as create_dataset handles splitting)
    full_df = data_preprocessing.load_specific_language_data(raw_data_path, lang_codes_to_load)
    if full_df.empty:
        print("No data loaded. Exiting.")
        return
    
    full_df = data_preprocessing.preprocess_text(full_df)
    _, eval_dataset_hf = data_preprocessing.create_dataset(full_df, model_name)
    
    if eval_dataset_hf is None or len(eval_dataset_hf) == 0:
        print("Evaluation dataset is empty. Exiting.")
        return
    print(f"Evaluation dataset size: {len(eval_dataset_hf)}")

    # --- Load Model with Adapter ---
    print(f"Loading base model {model_name}...")
    # First, load the base model structure
    model = get_model(model_name, num_labels) # This already adds and configures a task adapter
    
    print(f"Loading trained adapter from checkpoint: {checkpoint_path}")
    # Load the trained adapter weights into the 'emotion_classification' adapter
    # The head is automatically loaded if it was saved with the adapter.
    model.load_adapter(checkpoint_path, model_name_or_path=None, load_as="emotion_classification", set_active=True)
    model.to(device)
    model.eval() # Set model to evaluation mode

    print("Model and adapter loaded successfully.")

    # --- Generate Predictions ---
    print("Generating predictions...")
    all_logits = []
    
    # Manually iterate and predict to handle potential OOM with large eval sets if not using Trainer
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset_hf, batch_size=config.TRAINING_ARGS.get('per_device_eval_batch_size', 8))

    for batch in tqdm(eval_dataloader, desc="Predicting"):
        # Ensure batch items are tensors and on the correct device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        all_logits.append(outputs.logits.cpu().numpy())
    
    logits_np = np.concatenate(all_logits, axis=0)
    predictions_np = (logits_np > 0.5).astype(int) # Apply sigmoid and threshold for multilabel
    true_labels_np = np.array(eval_dataset_hf['labels'])

    # --- Compute Metrics ---
    print("\nComputing metrics...")
    # The compute_metrics function expects a PaddedTuple or similar structure
    # We'll create a simple structure that mimics it for our raw predictions and labels
    class EvalPrediction:
        def __init__(self, predictions, label_ids):
            self.predictions = predictions
            self.label_ids = label_ids
            
    eval_preds = EvalPrediction(predictions=logits_np, label_ids=true_labels_np)
    metrics = compute_metrics(eval_preds)

    # --- Display Metrics ---
    print("\n--- Evaluation Results ---")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Hamming Loss: {metrics['hamming_loss']:.4f}")
    print("\nPrecision:")
    print(f"  Micro:    {metrics['precision_micro']:.4f}")
    print(f"  Macro:    {metrics['precision_macro']:.4f}")
    print(f"  Weighted: {metrics['precision_weighted']:.4f}")
    print("\nRecall:")
    print(f"  Micro:    {metrics['recall_micro']:.4f}")
    print(f"  Macro:    {metrics['recall_macro']:.4f}")
    print(f"  Weighted: {metrics['recall_weighted']:.4f}")
    print("\nF1-Score:")
    print(f"  Micro:    {metrics['f1_micro']:.4f}")
    print(f"  Macro:    {metrics['f1_macro']:.4f}")
    print(f"  Weighted: {metrics['f1_weighted']:.4f}")
    print("-------------------------")

if __name__ == "__main__":
    main()

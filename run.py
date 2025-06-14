import sys
import argparse

print("--- SCRIPT EXECUTION VERIFICATION ---", file=sys.stderr)

# This is the main execution script for the data processing pipeline.
# It orchestrates the loading, preprocessing, and saving of the dataset.
import config
from src import data_preprocessing, model, train, evaluate, utils
import os
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, classification_report
from transformers import Trainer, TrainingArguments
from datasets import Dataset


def run_analysis():
    """Main function to run the error analysis pipeline."""
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

    print("Finding optimal prediction threshold...")
    best_f1 = 0
    best_threshold = 0.5
    for threshold in np.arange(0.1, 0.9, 0.01):
        binary_preds_loop = (probs > threshold).astype(int)
        _, _, f1_micro, _ = precision_recall_fscore_support(
            true_labels, binary_preds_loop, average="micro", zero_division=0
        )
        if f1_micro > best_f1:
            best_f1 = f1_micro
            best_threshold = threshold
    print(f"Optimal threshold found: {best_threshold:.2f} (F1 Micro: {best_f1:.4f})")
    binary_preds = (probs > best_threshold).astype(int)

    print("\n--- Classification Report ---")
    report = classification_report(
        true_labels, binary_preds, target_names=config.LABEL_COLUMNS, zero_division=0
    )
    print(report)

    true_labels_df = pd.DataFrame(
        true_labels, columns=[f"{col}_true" for col in config.LABEL_COLUMNS]
    )
    pred_labels_df = pd.DataFrame(
        binary_preds, columns=[f"{col}_pred" for col in config.LABEL_COLUMNS]
    )
    val_df.reset_index(drop=True, inplace=True)
    true_labels_df.reset_index(drop=True, inplace=True)
    pred_labels_df.reset_index(drop=True, inplace=True)
    review_df = pd.concat([val_df[["text"]], true_labels_df, pred_labels_df], axis=1)
    review_df["is_correct"] = (
        review_df[[f"{col}_true" for col in config.LABEL_COLUMNS]].values
        == review_df[[f"{col}_pred" for col in config.LABEL_COLUMNS]].values
    ).all(axis=1)
    output_path = os.path.join(config.OUTPUT_DIR, "error_analysis_results.csv")
    review_df.to_csv(output_path, index=False)
    print(f"\nSaved detailed predictions to: {output_path}")
    print(
        f"Number of incorrect predictions: {len(review_df[review_df['is_correct'] == False])}"
    )

    print("\n--- Error Analysis Pipeline Finished ---")


def main():
    """Main function to run the data processing pipeline."""
    # --- Argument Parsing ---
    # Set up an argument parser to provide help messages and allow for future command-line options.
    parser = argparse.ArgumentParser(
        description="Run the complete data processing and model training pipeline for the specified language."
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run error analysis on the validation set.",
    )
    # The parser automatically adds a -h/--help flag.
    args = parser.parse_args()

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

    print("Starting experiment pipeline...")

    # Ensure output directories exist
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.MODEL_OUTPUT_DIR, exist_ok=True)

    # Set a seed for reproducibility of results.
    utils.set_seed(config.TRAINING_ARGS["seed"])

    # --- Data Loading and Preprocessing ---
    # Define the path for the preprocessed data file and ensure the directory exists.
    processed_file_path = os.path.join(config.PROCESSED_DATA_PATH, "hau_processed.csv")
    os.makedirs(config.PROCESSED_DATA_PATH, exist_ok=True)

    # Check if a preprocessed file already exists to save time.
    if os.path.exists(processed_file_path):
        print(f"Found preprocessed data. Loading from: {processed_file_path}")
        df = pd.read_csv(processed_file_path)
    else:
        print("No preprocessed data found. Starting from raw data...")
        # Define the path to the raw dataset directory.
        raw_data_path = os.path.join(
            config.RAW_DATA_PATH,
            "SemEval2025-Task11",
            "task-dataset",
            "semeval-2025-task11-dataset",
            "track_a",
            "train",
        )

        # Load the dataset for the specified language(s).
        df = data_preprocessing.load_specific_language_data(
            raw_data_path, [config.LANG_CODE]
        )

        # Preprocess and save the data if it was loaded successfully.
        if not df.empty:
            df = data_preprocessing.preprocess_text(df)
            try:
                df.to_csv(processed_file_path, index=False, encoding="utf-8-sig")
                print(
                    f"Successfully processed and saved data to: {processed_file_path}"
                )
            except Exception as e:
                print(f"Could not save processed data due to an error: {e}")

    # Proceed with the rest of the pipeline only if data was successfully loaded/created.
    if "df" in locals() and not df.empty:
        # --- Model Initialization ---
        # Define the number of emotions we are classifying.
        num_labels = 6  # anger, disgust, fear, joy, sadness, surprise

        # Load the pre-trained model using the name specified in the config.
        # This function is defined in src/model.py.
        active_model = model.get_model(config.MODELS["afrixlmr"], num_labels)

        # --- Device Configuration ---
        # Set the device to GPU if available, otherwise CPU.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on device: {device}")
        active_model.to(device)

        # --- Dataset Creation and Class Weights ---
        # Tokenize the text and prepare the data for training.
        train_dataset, eval_dataset, label_columns = data_preprocessing.create_dataset(
            df, config.MODELS["afrixlmr"]
        )

        # Calculate class weights to handle data imbalance.
        # We need the original dataframe and the list of one-hot encoded label columns.
        class_weights = data_preprocessing.calculate_class_weights(df, label_columns)

        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(eval_dataset)}")

        # --- Train the Model ---
        # Pass the model, datasets, and class weights to the training function.
        trainer = train.train_model(active_model, train_dataset, eval_dataset, class_weights)

        # --- Final Evaluation ---
        if trainer:
            print("\n--- Final Model Evaluation ---")
            print("Evaluating the best model on the validation set...")
            final_metrics = trainer.evaluate()

            print("\n--- Overall Metrics ---")
            print(f"  Accuracy (Subset): {final_metrics.get('eval_accuracy_subset', 0):.4f}")
            print(f"  Hamming Loss:      {final_metrics.get('eval_hamming_loss', 0):.4f}")
            print(f"  F1-Score (Micro):    {final_metrics.get('eval_f1_micro', 0):.4f}")
            print(f"  F1-Score (Macro):    {final_metrics.get('eval_f1_macro', 0):.4f}")
            print(f"  F1-Score (Weighted): {final_metrics.get('eval_f1_weighted', 0):.4f}")
            print(f"  Precision (Micro):   {final_metrics.get('eval_precision_micro', 0):.4f}")
            print(f"  Precision (Macro):   {final_metrics.get('eval_precision_macro', 0):.4f}")
            print(f"  Precision (Weighted):{final_metrics.get('eval_precision_weighted', 0):.4f}")
            print(f"  Recall (Micro):      {final_metrics.get('eval_recall_micro', 0):.4f}")
            print(f"  Recall (Macro):      {final_metrics.get('eval_recall_macro', 0):.4f}")
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

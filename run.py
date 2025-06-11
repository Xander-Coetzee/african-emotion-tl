# This is the main execution script for the data processing pipeline.
# It orchestrates the loading, preprocessing, and saving of the dataset.
import config
from src import data_preprocessing, model, train, evaluate, utils
import os
import pandas as pd
import torch



def main():
    """Main function to run the data processing pipeline."""
    print("Starting experiment pipeline...")

    # Set a seed for reproducibility of results.
    utils.set_seed(config.TRAINING_ARGS["seed"])

    # Define the path to the raw dataset directory using the base path from config.
    raw_data_path = os.path.join(config.RAW_DATA_PATH, "SemEval2025-Task11", "task-dataset", "semeval-2025-task11-dataset", "track_a", "train")

    # Load the dataset for the language(s) specified in the config file.
    # This makes our pipeline configurable without changing the code.
    df = data_preprocessing.load_specific_language_data(raw_data_path, config.LANGUAGES)

    # Proceed with preprocessing only if data was successfully loaded.
    if not df.empty:
        # Apply text cleaning and normalization steps.
        df = data_preprocessing.preprocess_text(df)

        # Ensure the directory for processed data exists.
        processed_data_dir = config.PROCESSED_DATA_PATH
        os.makedirs(processed_data_dir, exist_ok=True)

        # Save a sample of the cleaned data to a CSV file for inspection.
        sample_path = os.path.join(processed_data_dir, "data_sample.csv")
        try:
            df.head(1000).to_csv(sample_path, index=False, encoding='utf-8-sig')
            print(f"\nSuccessfully processed data. A sample has been saved to: {sample_path}")
        except Exception as e:
            print(f"\nCould not save data sample due to an error: {e}")

        # --- Model Initialization ---
        # Define the number of emotions we are classifying.
        num_labels = 6  # anger, disgust, fear, joy, sadness, surprise

        # Load the pre-trained model using the name specified in the config.
        # This function is defined in src/model.py.
        active_model = model.get_model(config.MODELS['afrixlmr'], num_labels)

        # --- Device Configuration ---
        # Set the device to GPU if available, otherwise CPU.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on device: {device}")
        active_model.to(device)

        # --- Dataset Creation and Class Weights ---
        # Tokenize the text and prepare the data for training.
        train_dataset, eval_dataset, label_columns = data_preprocessing.create_dataset(df, config.MODELS['afrixlmr'])

        # Calculate class weights to handle data imbalance.
        # We need the original dataframe and the list of one-hot encoded label columns.
        class_weights = data_preprocessing.calculate_class_weights(df, label_columns)

        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(eval_dataset)}")

        # --- Train the Model ---
        # Pass the model, datasets, and class weights to the training function.
        train.train_model(active_model, train_dataset, eval_dataset, class_weights)

    print("\nExperiment pipeline finished.")


if __name__ == "__main__":
    main()

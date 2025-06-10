# This is the main execution script for the data processing pipeline.
# It orchestrates the loading, preprocessing, and saving of the dataset.
import config
from src import data_preprocessing, model, train, evaluate, utils


import os

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
        print(f"Model {config.MODELS['afrixlmr']} loaded successfully.")

        # --- Dataset Creation ---
        # Tokenize the text and prepare the data for training.
        # This function is defined in src/data_preprocessing.py
        train_dataset, eval_dataset = data_preprocessing.create_dataset(df, config.MODELS['afrixlmr'])

        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(eval_dataset)}")

        # --- Future steps for the pipeline will be added below ---
        # 4. Train the model
        # 5. Evaluate the model

    print("\nExperiment pipeline finished.")


if __name__ == "__main__":
    main()

# To manage all configurations and hyperparameters.

# Model configurations
MODELS = {
    "mbert": "bert-base-multilingual-cased",
    "xlm-roberta": "xlm-roberta-base",
    "afriberta": "castorini/afriberta_base",
    "afrixlmr": "Davlan/afro-xlmr-base",
}

# Specifies the ISO 639-1 code for the language to be processed.
# We are focusing only on Hausa as per the project requirements.
LANG_CODE = "hau"

# Define the names of the emotion labels being classified.
LABEL_COLUMNS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']

# Training parameters
# Define the model to use and the name for the adapter.
MODEL_NAME = MODELS['afrixlmr']
ADAPTER_NAME = "emotion_classification"

# Data splitting parameters
TRAIN_TEST_SPLIT_RATIO = 0.8
SEED = 42

# Training parameters
TRAINING_ARGS = {
    "learning_rate": 2e-5,
    "num_train_epochs": 20,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "weight_decay": 0.01,
    "seed": SEED,
}

# Data paths
DATA_PATH = "data/"
RAW_DATA_PATH = "data/brighter/brighter_data_train/"
PROCESSED_DATA_PATH = f"{DATA_PATH}processed/"
OUTPUT_DIR = "results/"
MODEL_OUTPUT_DIR = "models/"
SAVED_MODEL_PATH = MODEL_OUTPUT_DIR

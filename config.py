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
LANGUAGES = ["hau"]

# Training parameters
TRAINING_ARGS = {
    "learning_rate": 2e-5,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "weight_decay": 0.01,
    "seed": 42,
}

# Data paths
DATA_PATH = "data/"
RAW_DATA_PATH = f"{DATA_PATH}raw/"
PROCESSED_DATA_PATH = f"{DATA_PATH}processed/"

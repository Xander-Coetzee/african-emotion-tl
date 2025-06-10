# To define the different model architectures (fine-tuning, adapters).
from transformers import AutoModelForSequenceClassification


def get_model(model_name: str, num_labels: int):
    """Initializes a transformer model for sequence classification.

    This function leverages the Hugging Face Transformers library to load a pre-trained model.
    It's configured for sequence classification, making it suitable for our emotion analysis task.

    Args:
        model_name (str): The identifier of the pre-trained model to load (e.g., 'bert-base-uncased').
        num_labels (int): The number of distinct labels in the classification task.

    Returns:
        A transformer model instance ready for training.
    """
    # Announce which model is being loaded for clarity during execution.
    print(f"Loading model: {model_name}")

    # Load the pre-trained model, specifying the number of labels for the classification head.
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # Return the initialized model.
    return model

# To define the different model architectures (fine-tuning, adapters).
from adapters import AutoAdapterModel
from adapters.composition import Stack

def get_model(model_name: str, num_labels: int):
    """
    Loads a pre-trained transformer model and sets it up for adapter-based tuning.

    This function loads the base model, adds a new task-specific adapter for sequence
    classification, activates it, and freezes the base model's weights. This ensures
    that only the small adapter module is trained, which is much more efficient.

    Args:
        model_name (str): The identifier for the pre-trained model.
        num_labels (int): The number of labels for the classification task.

    Returns:
        A transformer model instance ready for training.
    """
    # Announce which model is being loaded for clarity during execution.
    print(f"Loading model: {model_name}")
    model = AutoAdapterModel.from_pretrained(model_name)

    # Add a new task-specific adapter for sequence classification.
    # We'll give it a unique name, 'emotion_classification'.
    model.add_adapter("emotion_classification", config="pfeiffer")

    # Add a classification head that matches our number of labels.
    model.add_classification_head("emotion_classification", num_labels=num_labels)

    # Set the active adapter for training.
    model.train_adapter("emotion_classification")

    return model

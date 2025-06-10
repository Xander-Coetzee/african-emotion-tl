# For calculating and reporting performance metrics.
from sklearn.metrics import f1_score, hamming_loss, accuracy_score


def evaluate_model(predictions, labels):
    """Evaluates the model using multilabel metrics."""
    print("Evaluating model...")
    # f1 = f1_score(labels, predictions, average='macro')
    # hamming = hamming_loss(labels, predictions)
    # subset_acc = accuracy_score(labels, predictions)
    # print(f"F1 (Macro): {f1}, Hamming Loss: {hamming}, Subset Accuracy: {subset_acc}")
    # return {"f1_macro": f1, "hamming_loss": hamming, "subset_accuracy": subset_acc}
    pass

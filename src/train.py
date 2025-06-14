# To handle the model training loops.
from adapters import AdapterTrainer
from transformers import TrainingArguments
import torch
from torch.nn import BCEWithLogitsLoss
import config
import math
from src.evaluate import compute_metrics
import os
import shutil


class MultilabelTrainer(AdapterTrainer):
    """Custom trainer for multi-label classification that handles class weights."""
    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        # Pass the class weights to the loss function.
        loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                        labels.view(-1, self.model.config.num_labels).float())
        return (loss, outputs) if return_outputs else loss

def train_model(model, train_dataset, eval_dataset, class_weights):
    """
    Sets up and runs the adapter-based training process.

    Args:
        model: The adapter-enabled model to be trained.
        train_dataset: The dataset for training.
        eval_dataset: The dataset for evaluation.
    """
    # Configure training arguments for a more robust training and evaluation cycle.
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=config.TRAINING_ARGS.get('num_train_epochs', 3), # Use epoch number from config.
        lr_scheduler_type="cosine",  # Use a cosine learning rate scheduler.
        per_device_train_batch_size=config.TRAINING_ARGS.get('per_device_train_batch_size', 8),
        per_device_eval_batch_size=config.TRAINING_ARGS.get('per_device_eval_batch_size', 8),
        warmup_steps=config.TRAINING_ARGS.get('warmup_steps', 500),
        weight_decay=config.TRAINING_ARGS.get('weight_decay', 0.01),
        logging_dir='./logs',
        logging_strategy="epoch", # Log metrics at the end of each epoch.
        eval_strategy="epoch", # Evaluate at the end of each epoch.
        save_strategy="epoch", # Save a checkpoint at the end of each epoch.
        load_best_model_at_end=True, # Load the best model when training is complete.
        metric_for_best_model="f1_micro", # Use f1_micro to determine the best model.
        greater_is_better=True # Higher f1_micro is better.
    )

    # Initialize the custom MultilabelTrainer with class weights.
    trainer = MultilabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics, # Pass in our metrics function.
        pos_weight=class_weights.to(model.device) # Pass class weights to the trainer.
    )

    # Start the training process.
    print("Starting adapter-based model training...")
    trainer.train()
    print("Model training complete.")

    # --- Save the Best Model & Clean Up ---
    # After training, the best model is loaded. We save the adapter and the head.
    adapter_name = "emotion_classification"  # This should match the adapter name used in model.py
    
    # Ensure the save directory exists.
    os.makedirs(config.SAVED_MODEL_PATH, exist_ok=True)
    
    # Define the full path for the saved adapter.
    final_adapter_path = os.path.join(config.SAVED_MODEL_PATH, adapter_name)

    # Save the adapter and the prediction head.
    model.save_adapter(final_adapter_path, adapter_name)
    model.save_head(final_adapter_path, adapter_name)
    
    print(f"Best adapter and head saved to {final_adapter_path}")

    # Clean up the checkpoints directory.
    print(f"Cleaning up checkpoint directory: {training_args.output_dir}")
    shutil.rmtree(training_args.output_dir)
    print("Cleanup complete.")

    return trainer

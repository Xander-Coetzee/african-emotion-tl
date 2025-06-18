# To handle the model training loops.
from adapters import AdapterTrainer
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
import config
import math
from src.evaluate import compute_metrics
from src.visualization import plot_training_metrics
import os
import shutil


def get_focal_loss_trainer():
    """
    Create a custom trainer with focal loss for better precision/recall balance.
    """
    class FocalLoss(torch.nn.Module):
        def __init__(self, alpha=1, gamma=2, reduction='mean'):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction
        
        def forward(self, inputs, targets):
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            pt = torch.exp(-bce_loss)
            focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
            
            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            return focal_loss

    class FocalLossTrainer(AdapterTrainer):
        def __init__(self, *args, **kwargs):
            # Remove pos_weight from kwargs if it's there, as we are not using it with focal loss
            kwargs.pop('pos_weight', None)
            super().__init__(*args, **kwargs)
            self.focal_loss = FocalLoss(alpha=1, gamma=2)
        
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            # Pop labels from inputs to prevent the model from calculating loss internally.
            # We will calculate it manually using our custom focal loss.
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.get('logits')

            loss = self.focal_loss(logits, labels.float())

            return (loss, outputs) if return_outputs else loss

    return FocalLossTrainer

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
        logging_dir='./results',  # Save logs in results directory
        logging_strategy="epoch", # Log metrics at the end of each epoch.
        eval_strategy="epoch", # Evaluate at the end of each epoch.
        save_strategy="epoch", # Save a checkpoint at the end of each epoch.
        save_total_limit=3,  # Keep the best 3 models
        load_best_model_at_end=True, # Load the best model when training is complete.
        metric_for_best_model="f1_micro", # Use f1_micro to determine the best model.
        greater_is_better=True, # Higher f1_micro is better.
        report_to=None,  # Disable other logging to avoid conflicts
        save_safetensors=False,  # Save as .bin for compatibility
        save_on_each_node=True  # Ensure saving works in distributed training
    )

    # Initialize the FocalLossTrainer.
    # Note: We are replacing the pos_weight mechanism with Focal Loss.
    TrainerClass = get_focal_loss_trainer()
    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics, # Pass in our metrics function.
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

    # Generate and save training metrics plots
    print("\nGenerating training metrics plots...")
    plot_training_metrics(
        output_dir=os.path.join(training_args.output_dir, 'plots')
    )
    
    # Clean up the checkpoints directory but keep the logs for visualization
    print(f"\nCleaning up checkpoint directory: {training_args.output_dir}")
    # Remove only the checkpoint directories, keep the logs
    for item in os.listdir(training_args.output_dir):
        if item.startswith('checkpoint-'):
            shutil.rmtree(os.path.join(training_args.output_dir, item))
    print("Cleanup complete.")

    return trainer

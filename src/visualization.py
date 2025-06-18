"""
Visualization utilities for training metrics.
"""
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_training_metrics(output_dir='./results/plots'):
    """
    Plot training and evaluation metrics from the training logs.
    
    Args:
        output_dir (str): Directory to save the plots and look for logs
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Possible locations for the trainer state file
    possible_paths = [
        Path('./results/trainer_state.json'),
        Path('./results/checkpoint-*/trainer_state.json'),
        Path('./trainer_state.json'),
    ]
    
    log_file = None
    for path in possible_paths:
        matches = list(path.parent.glob(path.name)) if '*' in str(path) else [path]
        for match in matches:
            if match.exists():
                log_file = match
                break
        if log_file is not None:
            break
    
    if log_file is None or not log_file.exists():
        print("Could not find trainer_state.json in any of the expected locations:")
        for path in possible_paths:
            print(f"- {path}")
        print("\nPlease ensure the training has completed successfully.")
        return
    
    print(f"Found training logs at: {log_file.absolute()}")
    
    try:
        # Load the log file
        with open(log_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Failed to parse {log_file}: {e}")
        return
    
    # Extract metrics
    metrics = {}
    for entry in logs['log_history']:
        for key, value in entry.items():
            if key.startswith(('train_', 'eval_')):
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(value)
    
    if not metrics:
        print("No metrics found in logs")
        return
    
    # Plot training and evaluation loss
    if 'train_loss' in metrics and 'eval_loss' in metrics:
        plt.figure(figsize=(10, 5))
        plt.plot(metrics['train_loss'], label='Training Loss')
        plt.plot(metrics['eval_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/loss_plot.png")
        plt.close()
    
    # Plot F1 scores if available
    f1_metrics = {k: v for k, v in metrics.items() if 'f1' in k and '_f1' not in k}
    if f1_metrics:
        plt.figure(figsize=(12, 6))
        for metric, values in f1_metrics.items():
            plt.plot(values, label=metric.replace('eval_', '').replace('_', ' ').title())
        plt.title('F1 Scores')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/f1_scores.png")
        plt.close()
    
    # Plot precision and recall
    precision_metrics = {k: v for k, v in metrics.items() if 'precision' in k and '_precision' not in k}
    recall_metrics = {k: v for k, v in metrics.items() if 'recall' in k and '_recall' not in k}
    
    if precision_metrics:
        plt.figure(figsize=(12, 6))
        for metric, values in precision_metrics.items():
            plt.plot(values, '--', label=f"{metric.replace('eval_', '').replace('_', ' ').title()}")
        plt.title('Precision Scores')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/precision_scores.png")
        plt.close()
    
    if recall_metrics:
        plt.figure(figsize=(12, 6))
        for metric, values in recall_metrics.items():
            plt.plot(values, '-', label=f"{metric.replace('eval_', '').replace('_', ' ').title()}")
        plt.title('Recall Scores')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/recall_scores.png")
        plt.close()
    
    print(f"Plots saved to {output_dir}")

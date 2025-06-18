# African Emotion Transfer Learning

A deep learning project for emotion classification in African languages, focusing on transfer learning with pre-trained language models and adapter-based fine-tuning.
## Project Structure

```
african-emotion-tl/
├── config.py           # Configuration parameters and paths
├── run.py             # Main script for training and evaluation
├── requirements.txt   # Project dependencies
├── data/              # Data directory
│   └── raw/           # Raw dataset files
├── models/            # Saved model checkpoints
├── results/           # Training logs and evaluation results
└── src/               # Source code
    ├── data_preprocessing.py  # Data loading and preprocessing
    ├── model.py              # Model architecture
    ├── train.py              # Training loop
    ├── evaluate.py           # Evaluation metrics
    ├── visualization.py      # Plotting utilities
    └── utils.py              # Helper functions
```
## Prerequisites
- dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/african-emotion-tl.git
   cd african-emotion-tl
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your data:
   - Place your dataset in the `data/raw/SemEval2025-Task11/task-dataset/semeval-2025-task11-dataset/track_a/train/` directory
   - Ensure the data follows the expected format (see Data Format section)

## Usage

### Training a Model

Train a model with default settings:
```bash
python run.py --lang hau --model xlm-roberta-base --epochs 20 --batch-size 16
```

### Command Line Arguments

- `--model`: Model to use (default: xlm-roberta-base)
- `--lang`: Language code (default: hau)
- `--epochs`: Number of training epochs (default: 20)
- `--batch-size`: Batch size for training/evaluation (default: 16)
- `--learning-rate`: Learning rate (default: 2e-5)
- `--output-dir`: Directory to save results (default: results/)
- `--seed`: Random seed (default: 42)


## Acknowledgments

- [Hugging Face](https://huggingface.co/) for the Transformers library
- [AdapterHub](https://adapterhub.ml/) for the adapter-transformers library
- The BRIGHTER dataset for the emotion classification task

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ==============================================================================
# STEP 1: PARSED EXPERIMENT DATA
# I have manually parsed the data you provided from your terminal logs.
# You do not need to change anything here.
# ==============================================================================

# Data from "English with Afro-XLM"
eng_afroxlm = {
    "model": "AfriBERTa ",
    "language": "English",
    "accuracy_subset": 0.3396,
    "hamming_loss": 0.1634,
    "f1_micro": 0.5945,
    "f1_macro": 0.3665,
    "f1_weighted": 0.5603,
    "precision_macro": 0.6036,
    "recall_macro": 0.3383,
    "precision_anger": 1.0000,
    "recall_anger": 0.0081,
    "f1_anger": 0.0160,
    "precision_disgust": 0.0000,
    "recall_disgust": 0.0000,
    "f1_disgust": 0.0000,
    "precision_fear": 0.6911,
    "recall_fear": 0.7873,
    "f1_fear": 0.7361,
    "precision_joy": 0.6209,
    "recall_joy": 0.5528,
    "f1_joy": 0.5849,
    "precision_sadness": 0.5795,
    "recall_sadness": 0.4424,
    "f1_sadness": 0.5018,
    "precision_surprise": 0.7303,
    "recall_surprise": 0.2392,
    "f1_surprise": 0.3604,
}

# Data from "English with XLM-RoBERTa" (Parsed from the messy log at epoch 20.0)
eng_xlmroberta = {
    "model": "XLM-RoBERTa",
    "language": "English",
    "accuracy_subset": 0.3995,
    "hamming_loss": 0.1397,
    "f1_micro": 0.6783,
    "f1_macro": 0.4936,
    "f1_weighted": 0.6669,
    "precision_macro": 0.6090,
    "recall_macro": 0.4572,
    "precision_anger": 0.8571,
    "recall_anger": 0.1818,
    "f1_anger": 0.3000,
    "precision_disgust": 0.0000,
    "recall_disgust": 0.0000,
    "f1_disgust": 0.0000,
    "precision_fear": 0.7256,
    "recall_fear": 0.7944,
    "f1_fear": 0.7584,
    "precision_joy": 0.7522,
    "recall_joy": 0.6640,
    "f1_joy": 0.7053,
    "precision_sadness": 0.6528,
    "recall_sadness": 0.6076,
    "f1_sadness": 0.6294,
    "precision_surprise": 0.6666,
    "recall_surprise": 0.4955,
    "f1_surprise": 0.5685,
}

# Data from "Hausa with Afro-XLM"
hau_afroxlm = {
    "model": "AfriBERTa ",
    "language": "Hausa",
    "accuracy_subset": 0.3252,
    "hamming_loss": 0.1518,
    "f1_micro": 0.4964,
    "f1_macro": 0.4161,
    "f1_weighted": 0.4366,
    "precision_macro": 0.7327,
    "recall_macro": 0.3919,
    "precision_anger": 0.5698,
    "recall_anger": 0.3400,
    "f1_anger": 0.4259,
    "precision_disgust": 0.8714,
    "recall_disgust": 0.4880,
    "f1_disgust": 0.6256,
    "precision_fear": 0.5435,
    "recall_fear": 0.7202,
    "f1_fear": 0.6195,
    "precision_joy": 0.8438,
    "recall_joy": 0.1059,
    "f1_joy": 0.1882,  # Calculated F1
    "precision_sadness": 0.5677,
    "recall_sadness": 0.6899,
    "f1_sadness": 0.6229,
    "precision_surprise": 1.0000,
    "recall_surprise": 0.0074,
    "f1_surprise": 0.0148,
}

# Data from "Hausa with XLM-RoBERTa"
hau_xlmroberta = {
    "model": "XLM-RoBERTa",
    "language": "Hausa",
    "accuracy_subset": 0.3610,
    "hamming_loss": 0.1421,
    "f1_micro": 0.5232,
    "f1_macro": 0.4833,
    "f1_weighted": 0.4914,
    "precision_macro": 0.6104,
    "recall_macro": 0.4512,
    "precision_anger": 0.4800,
    "recall_anger": 0.3582,
    "f1_anger": 0.4103,
    "precision_disgust": 0.6462,
    "recall_disgust": 0.6462,
    "f1_disgust": 0.6462,
    "precision_fear": 0.5370,
    "recall_fear": 0.6170,
    "f1_fear": 0.5743,
    "precision_joy": 0.8276,
    "recall_joy": 0.3333,
    "f1_joy": 0.4752,
    "precision_sadness": 0.6261,
    "recall_sadness": 0.6667,
    "f1_sadness": 0.6457,
    "precision_surprise": 0.5455,
    "recall_surprise": 0.0857,
    "f1_surprise": 0.1481,
}

# --- Group the data for easy plotting ---
hausa_results = [hau_afroxlm, hau_xlmroberta]
english_results = [eng_afroxlm, eng_xlmroberta]

# ==============================================================================
# STEP 2: PLOTTING FUNCTIONS
# You don't need to change anything in these functions.
# ==============================================================================


def plot_overall_metrics(results_data, language, output_dir="plots"):
    """
    Generates a grouped bar chart comparing overall performance metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(results_data)

    metrics_to_plot = {
        "accuracy_subset": "Accuracy",
        "f1_macro": "F1 (Macro)",
        "precision_macro": "Precision",
        "recall_macro": "Recall",
    }

    plot_data = df.set_index("model")[metrics_to_plot.keys()].rename(
        columns=metrics_to_plot
    )

    ax = plot_data.plot(kind="bar", figsize=(12, 7), width=0.8, colormap="viridis")

    plt.title(f"Overall Model Performance Comparison on {language}", fontsize=16)
    plt.ylabel("Score", fontsize=12)
    plt.xlabel("")  # Remove x-axis label for cleaner look
    plt.xticks(rotation=0, fontsize=12)
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(title="Metric", fontsize=11)

    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 9),
            textcoords="offset points",
            fontsize=9,
            color="dimgray",
        )

    filename = os.path.join(output_dir, f"overall_metrics_{language.lower()}.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"✅ Saved overall metrics plot to: {filename}")
    plt.show()


def plot_per_emotion_metrics(model_result, output_dir="plots"):
    """
    Generates a grouped bar chart for per-emotion Precision, Recall, and F1.
    """
    os.makedirs(output_dir, exist_ok=True)
    model_name = model_result["model"]
    language = model_result["language"]
    emotions = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

    precision = [model_result[f"precision_{e}"] for e in emotions]
    recall = [model_result[f"recall_{e}"] for e in emotions]
    f1 = [model_result[f"f1_{e}"] for e in emotions]

    x = np.arange(len(emotions))
    width = 0.25
    fig, ax = plt.subplots(figsize=(14, 8))

    rects1 = ax.bar(x - width, precision, width, label="Precision")
    rects2 = ax.bar(x, recall, width, label="Recall")
    rects3 = ax.bar(x + width, f1, width, label="F1")

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Per-Emotion Performance of {model_name} on {language}", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([e.capitalize() for e in emotions], fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.legend(fontsize=11)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()

    filename = os.path.join(
        output_dir, f'per_emotion_{model_name.replace("-","_")}_{language.lower()}.png'
    )
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"✅ Saved per-emotion metrics plot to: {filename}")
    plt.show()


# ==============================================================================
# STEP 3: GENERATE YOUR GRAPHS
# This will create and save all the plots you need.
# ==============================================================================
if __name__ == "__main__":
    # --- Generate Overall Comparison Plots (One for Hausa, one for English) ---
    print("\n--- Generating Overall Metrics Plots ---")
    plot_overall_metrics(hausa_results, language="Hausa")
    plot_overall_metrics(english_results, language="English")

    # --- Generate Per-Emotion Breakdown Plots (One for each of the 4 experiments) ---
    print("\n--- Generating Per-Emotion Metrics Plots ---")
    for result in hausa_results:
        plot_per_emotion_metrics(result)
    for result in english_results:
        plot_per_emotion_metrics(result)

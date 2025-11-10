import os
import torch
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt


def evaluate(model, loader, device):
    """
    Evaluate the model on the validation/test set and return accuracy and F1-score
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # get predictions
            outputs = model(X_batch).squeeze(1)
            all_preds.extend(outputs.cpu().numpy().tolist())
            all_labels.extend(y_batch.cpu().numpy().tolist())

    # convert probabilities to 0 or 1
    binary_preds = [1 if p >= 0.5 else 0 for p in all_preds]

    # calculate metrics
    acc = accuracy_score(all_labels, binary_preds)
    f1 = f1_score(all_labels, binary_preds, average="macro")

    return acc, f1


def generate_plots():
    """
    Creates two plots: Accuracy vs Sequence Length and F1 vs Sequence Length and Training Loss vs Epochs for the best and worst models
    """
    # check if results file exists
    if not os.path.exists("results/metrics.csv"):
        print("Error: results/metrics.csv not found!")
        return

    # load the results
    df = pd.read_csv("results/metrics.csv")
    os.makedirs("results/plots", exist_ok=True)

    # Plot 1: Accuracy and F1 vs Sequence Length
    # Filter for: LSTM, ReLU, Adam, no clipping
    seq_data = df[(df["arch"] == "lstm") & (df["activation"] == "relu") & (df["optimizer"] == "adam") & (df["grad_clip"] == False)].sort_values("seq_len")

    if len(seq_data) > 0:
        plt.figure(figsize=(12, 5))
        
        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(seq_data["seq_len"], seq_data["accuracy"], marker="o", linewidth=2)
        plt.xlabel("Sequence Length")
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs Sequence Length")
        plt.grid(True)
        
        # F1 plot
        plt.subplot(1, 2, 2)
        plt.plot(seq_data["seq_len"], seq_data["f1"], marker="o", linewidth=2)
        plt.xlabel("Sequence Length")
        plt.ylabel("F1 Score")
        plt.title("F1 Score vs Sequence Length")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("results/plots/metrics_vs_seqlen.png", dpi=300)
        plt.close()

    # Plot 2: Training Loss for best and worst models
    best_idx = df["accuracy"].idxmax()
    worst_idx = df["accuracy"].idxmin()
    
    best_model = df.loc[best_idx]
    worst_model = df.loc[worst_idx]
    
    # parse the loss strings
    best_losses = [float(x) for x in best_model["losses"].split(",")]
    worst_losses = [float(x) for x in worst_model["losses"].split(",")]
    
    epochs = list(range(1, len(best_losses) + 1))
    
    plt.figure(figsize=(10, 5))
    
    # create labels
    best_label = f"Best: {best_model['arch']}, {best_model['activation']}, {best_model['optimizer']}"
    worst_label = f"Worst: {worst_model['arch']}, {worst_model['activation']}, {worst_model['optimizer']}"
    
    plt.plot(epochs, best_losses, marker="o", label=best_label)
    plt.plot(epochs, worst_losses, marker="s", label=worst_label)
    
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss: Best vs Worst Model")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig("results/plots/loss_comparison.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    generate_plots()
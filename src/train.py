import time
import argparse
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from preprocess import prepare_data
from models import SentimentModel
from evaluate import evaluate
from utils import set_seed


def train_one_epoch(model, loader, criterion, optimizer, device, clip_grad=False):
    """
    Train the model for one epoch and return the average loss for this epoch
    """
    model.train()
    total_loss = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).float()

        # forward pass
        predictions = model(X_batch).squeeze(1)
        loss = criterion(predictions, y_batch)

        # backward pass
        optimizer.zero_grad()
        loss.backward()

        # gradient clipping if enabled
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    return avg_loss


def main(args):
    """
    Main training function
    """
    set_seed(42)

    # checking if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # loading and preprocessing data
    X_train, y_train, X_test, y_test, vocab = prepare_data(seq_len=args.seq_len)

    # converting to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.long)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # creating data loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # creating the model
    model = SentimentModel(vocab_size=len(vocab), rnn_type=args.arch, activation=args.activation, dropout=0.5).to(device)
    print(model)

    # choosing optimizer
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=0.01)
    elif args.optimizer == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    else:
        raise ValueError("Invalid optimizer")

    # loss function
    criterion = nn.BCELoss()

    # training loop
    start_time = time.time()
    epoch_times = []
    train_losses = []

    for epoch in range(args.epochs):
        epoch_start = time.time()

        # training for one epoch
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, clip_grad=args.clip
        )
        train_losses.append(train_loss)

        # evaluating on test set
        acc, f1 = evaluate(model, test_loader, device)
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {train_loss:.4f} | "
              f"Acc: {acc:.4f} | F1: {f1:.4f} | Time: {epoch_time:.2f}s")

    total_time = time.time() - start_time
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    
    print(f"\nTraining completed in {total_time/60:.2f} minutes")
    print(f"Final test accuracy: {acc:.4f}, F1-score: {f1:.4f}")

    os.makedirs("results", exist_ok=True)
    csv_path = "results/metrics.csv"
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow(["arch", "activation", "optimizer", "seq_len", "grad_clip", "accuracy", "f1", "Epoch Time(s)", "losses"])
        
        # saving losses as comma-separated string
        losses_str = ",".join([f"{loss:.4f}" for loss in train_losses])
        
        writer.writerow([
            args.arch, args.activation, args.optimizer, args.seq_len,
            args.clip, acc, f1, avg_epoch_time, losses_str
        ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RNN/LSTM sentiment model")

    parser.add_argument("--arch", type=str, default="lstm", help="Architecture: rnn | lstm | bi_lstm")
    parser.add_argument("--activation", type=str, default="relu", help="Activation: relu | tanh | sigmoid")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer: adam | sgd | rmsprop")
    parser.add_argument("--seq_len", type=int, default=50, help="Sequence length: 25 | 50 | 100")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--clip", action="store_true", help="Enable gradient clipping")

    args = parser.parse_args()
    main(args)
# Comparative Analysis of RNN Architectures for Sentiment Classification

Sentiment classification is a core NLP task that categorizes text into emotional tones (positive/negative). This project systematically compares different RNN architectures, activation functions, optimizers, and other hyperparameters to identify the optimal configuration for sentiment analysis.

## Requirements

### Python Version
- Python 3.8+

### Dependencies
```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
nltk>=3.8.0
```

### Installation
```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── data/
│   └── IMDB Dataset.csv
├── src/
│   ├── evaluate.py         # Model Evaluation and plotting 
│   ├── preprocess.py       # Data preprocessing and tokenization
│   ├── models.py           # RNN/LSTM/Bi-LSTM model architectures
│   ├── train.py            # Model Training and metrics calculation 
│   ├── experiments.py      # Automated script to run all combinations of experiments
│   └── utils.py            # Utility functions
├── results/
│   ├── metrics.csv         # Experimental results
│   └── plots/              # Generated plots
├── report.pdf              # Project report
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Usage

### 1. Data Preprocessing
The preprocessing is automatically handled when training is run, but it can be tested separately:
```bash
python src/preprocess.py
```
This will show statistics for all three sequence lengths (25, 50, 100).

### 2. Training a Model
Train a single model configuration:
```bash
python src/train.py --arch lstm --activation relu --optimizer adam --seq_len 50 --epochs 5
```

#### Available Arguments:
- **--arch:** Model architecture (rnn, lstm, bi_lstm)
- **--activation:** Activation function (relu, tanh, sigmoid)
- **--optimizer:** Optimizer (adam, sgd, rmsprop)
- **--seq_len:** Sequence length (25, 50, 100)
- **--epochs:** Number of training epochs (default: 5)
- **--clip:** Enable gradient clipping

#### Examples:
```bash
# Training LSTM with ReLU and Adam
python src/train.py --arch lstm --activation relu --optimizer adam --seq_len 50 --epochs 5

# Training Bi-LSTM with gradient clipping
python src/train.py --arch bi_lstm --activation tanh --optimizer adam --seq_len 100 --epochs 5 --clip

# Training RNN with SGD
python src/train.py --arch rnn --activation sigmoid --optimizer sgd --seq_len 25 --epochs 5
```

### 3. Running All Experiments

To systematically test all configurations, run this command:
```bash
python src/experiments.py
```
experiments.py script will systematically test all combinations of architectures, activations, optimizers, sequence lengths, and gradient clipping settings.

### 4. Generating Plots
After training multiple models, generate comparison plots:
```bash
python src/evaluate.py
```

This creates:

- **results/plots/metrics_vs_seqlen.png** - Accuracy/F1 vs Sequence Length

- **results/plots/loss_comparison.png** - Training Loss across 5 epochs: Best vs Worst Model

## Expected Runtime

All experiments use 5 epochs. Approximate times on CPU:

- **seq_len=25**: ~0.6 minutes 
- **seq_len=50**: ~1 minutes 
- **seq_len=100**: ~2 minutes 
- **Complete experiment suite**: ~50-60 minutes 

## Output Files

### metrics.csv
Contains all experimental results with columns: arch, activation, optimizer, seq_len, grad_clip, accuracy, f1, Epoch Time(s) & losses

### Generated Plots
- **metrics_vs_seqlen.png**: Shows how accuracy and F1-score change with sequence length
- **loss_comparison.png**: Compares training loss curves for best and worst models

## Troubleshooting

### Common Issues

**Issue**: FileNotFoundError: data/IMDB Dataset.csv
- **Solution**: Download the IMDB dataset and place it in the data/ folder

**Issue**: Missing NLTK data
- **Solution**: Run the following command:
```bash 
python -c "import nltk; nltk.download('punkt')"
```

## Reference

IMDB Dataset from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

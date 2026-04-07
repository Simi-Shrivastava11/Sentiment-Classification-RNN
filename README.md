# Comparative Analysis of RNN Architectures for Sentiment Classification

This project presents a comparative study of recurrent neural network architectures for **binary sentiment classification** on the **IMDB movie reviews dataset**. It evaluates how architecture choice, optimizer, activation function, sequence length, and gradient clipping affect model performance in a controlled experimental setup.

Developed as an **individual NLP/deep learning project**, this repository combines practical implementation with systematic experimentation to better understand the tradeoffs between different sequence models.

## Overview

Sentiment classification is a fundamental NLP task in which text is categorized according to emotional tone, such as **positive** or **negative** sentiment. In this project, I implemented and compared multiple recurrent architectures to identify which configurations work best for text classification under different training conditions.

The project compares:

- **Vanilla RNN**
- **LSTM**
- **Bidirectional LSTM**

Across multiple experimental settings, including:

- activation function
- optimizer
- sequence length
- gradient clipping

The goal was not just to train a single good model, but to **systematically compare model behavior** and understand which design choices have the greatest effect on performance.

## Why This Project Matters

This project demonstrates hands-on experience with:

- sequence modeling for NLP
- comparative experimentation in deep learning
- training stability analysis
- hyperparameter sensitivity
- reproducible evaluation pipelines

It also reflects a practical machine learning mindset: going beyond reporting one accuracy number and instead studying *why* certain model choices perform better than others.

## Features

- Implementation of **RNN**, **LSTM**, and **Bi-LSTM** architectures
- Comparison across **multiple activation functions**
- Comparison across **multiple optimizers**
- Experiments with **different sequence lengths**
- Optional **gradient clipping**
- Automated experiment runner for testing all configurations
- Evaluation pipeline with metrics and visualizations

## Methodology

### 1. Data Preprocessing

The project uses the **IMDB Dataset** for binary sentiment classification. Text preprocessing is handled automatically during training and can also be tested separately.

### 2. Model Architectures

The repository includes implementations of:

- **RNN**
- **LSTM**
- **Bi-LSTM**

These models are trained under consistent experimental settings to allow fair comparison.

### 3. Experimental Variables

The following parameters are varied across runs:

- **Architecture**: `rnn`, `lstm`, `bi_lstm`
- **Activation function**: `relu`, `tanh`, `sigmoid`
- **Optimizer**: `adam`, `sgd`, `rmsprop`
- **Sequence length**: `25`, `50`, `100`
- **Gradient clipping**: enabled or disabled

### 4. Evaluation

The project tracks performance using:

- **accuracy**
- **F1 score**
- **training loss**
- **epoch runtime**

It also generates plots to compare:
- performance vs. sequence length
- loss curves for best and worst models

## Tech Stack

- **Python**
- **PyTorch**
- **NumPy**
- **Pandas**
- **scikit-learn**
- **Matplotlib**
- **NLTK**

## Installation

### Requirements

- Python 3.8+

### Dependencies

Install the required packages with:

```bash
pip install -r requirements.txt
```

## How to Run

### 1. Data Preprocessing

Preprocessing is automatically handled during training, but it can also be tested separately:

```bash
python src/preprocess.py
```

This prints statistics for all supported sequence lengths.

### 2. Train a Single Model

Example:

```bash
python src/train.py --arch lstm --activation relu --optimizer adam --seq_len 50 --epochs 5
```

#### Available Arguments

- `--arch`: model architecture (`rnn`, `lstm`, `bi_lstm`)
- `--activation`: activation function (`relu`, `tanh`, `sigmoid`)
- `--optimizer`: optimizer (`adam`, `sgd`, `rmsprop`)
- `--seq_len`: sequence length (`25`, `50`, `100`)
- `--epochs`: number of training epochs (default: `5`)
- `--clip`: enable gradient clipping

#### More Examples

Train Bi-LSTM with gradient clipping:

```bash
python src/train.py --arch bi_lstm --activation tanh --optimizer adam --seq_len 100 --epochs 5 --clip
```

Train RNN with SGD:

```bash
python src/train.py --arch rnn --activation sigmoid --optimizer sgd --seq_len 25 --epochs 5
```

### 3. Run All Experiments

To test all configurations systematically:

```bash
python src/experiments.py
```

This script runs combinations of architectures, activations, optimizers, sequence lengths, and clipping settings.

### 4. Generate Plots

After training multiple models, generate comparison plots with:

```bash
python src/evaluate.py
```

This creates:

- `results/plots/metrics_vs_seqlen.png` — Accuracy and F1 score vs. sequence length
- `results/plots/loss_comparison.png` — Training loss comparison for best vs. worst models

## Expected Runtime

All experiments use 5 epochs. Approximate runtimes on CPU:

- `seq_len=25`: ~0.6 minutes
- `seq_len=50`: ~1 minute
- `seq_len=100`: ~2 minutes
- complete experiment suite: ~50–60 minutes

## Output Files

### `metrics.csv`

Contains all experiment results, including:

- architecture
- activation
- optimizer
- sequence length
- gradient clipping
- accuracy
- F1 score
- epoch time
- training loss

### Generated Plots

- `metrics_vs_seqlen.png`: shows how accuracy and F1 change with sequence length
- `loss_comparison.png`: compares training loss curves for best and worst models

## What I Learned

Through this project, I strengthened my understanding of:

- recurrent architectures for NLP
- how optimizers affect training performance
- the role of sequence length in sentiment modeling
- gradient clipping for training stability
- designing reproducible experiment pipelines in deep learning

## Future Improvements

Potential next steps include:

- adding pretrained embeddings such as GloVe or FastText
- comparing results against Transformer-based baselines
- performing deeper error analysis on misclassified reviews
- tracking experiments with a logging framework
- extending the evaluation to include precision/recall curves and confusion matrices

## Troubleshooting

### Common Issues

**Issue:** `FileNotFoundError: data/IMDB Dataset.csv`  
**Solution:** Download the IMDB dataset and place it in the `data/` folder.

**Issue:** Missing NLTK data  
**Solution:** Run:

```bash
python -c "import nltk; nltk.download('punkt')"
```

## Reference

IMDB Dataset from [Kaggle](https://www.kaggle.com)


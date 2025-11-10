import os
import re
from collections import Counter
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from utils import set_seed

# some global settings for the vocabulary
max_vocab_size = 10_000
pad_index = 0      # id for padding
unk_index = 1      # id for unknown words

def clean_text(text: str) -> str:
    """
    Basic text cleaning: lowercase, removing html tags, keeping simple chars, removing extra spaces
    """
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)              
    text = re.sub(r"[^a-z0-9\s.,!?']", " ", text)   
    text = re.sub(r"\s+", " ", text).strip()       
    return text

def build_vocab(tokenized_texts: List[List[str]],max_vocab_size: int = max_vocab_size) -> Dict[str, int]:
    """
    Build a vocabulary from a list of tokenized texts
    Reserved indices 0 for <pad> and 1 for <unk> and then assigned ids to the most frequent words.
    """
    counter = Counter()

    for tokens in tokenized_texts:
        counter.update(tokens)

    # most_common returns (word, count) pairs
    most_common = counter.most_common(max_vocab_size - 2)

    vocab = {"<pad>": pad_index, "<unk>": unk_index}
    for i, (word, _) in enumerate(most_common, start=2):
        vocab[word] = i

    return vocab

def tokens_to_ids(tokens: List[str], vocab: Dict[str, int], seq_len: int) -> List[int]:
    """
    Convert a token list into a fixed-length sequence of ids
    Unknown words mapped to <unk>, truncate if longer than seq_len, pad with <pad> if shorter
    """
    ids = [vocab.get(tok, unk_index) for tok in tokens]

    if len(ids) >= seq_len:
        ids = ids[:seq_len]
    else:
        # pad with pad_index up to seq_len
        ids = ids + [pad_index] * (seq_len - len(ids))

    return ids

def labels_to_int(sentiment_series: pd.Series) -> np.ndarray:
    """
    Convert the 'sentiment' column to integers 0/1
    """
    if sentiment_series.dtype == object:
        mapping = {"positive": 1, "negative": 0}
        return sentiment_series.str.lower().map(mapping).astype(int).values
    else:
        return sentiment_series.astype(int).values


def prepare_data(seq_len: int, data_path: str = "data/IMDB Dataset.csv") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """
    The dataset is a single CSV with 50,000 rows and columns: 'review' and 'sentiment'
    The predefined 50/50 split: first half = training set, second half = test set (no random shuffling before the split)

    Steps:
      1. Load the CSV.
      2. Split into train/test by index.
      3. Clean and tokenize reviews.
      4. Build vocab only on training tokens.
      5. Convert to padded/truncated id sequences.
      6. Turn sentiments into 0/1 labels.
    """
    set_seed(42)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find dataset at {data_path}")

    df = pd.read_csv(data_path)

    n_rows = len(df)
    half = n_rows // 2

    # predefined 50/50 split: first half = train, second half = test
    train_df = df.iloc[:half].reset_index(drop=True)
    test_df = df.iloc[half:].reset_index(drop=True)

    # clean text
    train_texts = [clean_text(t) for t in train_df["review"].astype(str)]
    test_texts = [clean_text(t) for t in test_df["review"].astype(str)]

    # tokenize
    train_tokens = [word_tokenize(t) for t in train_texts]
    test_tokens = [word_tokenize(t) for t in test_texts]

    # labels -> 0/1
    y_train = labels_to_int(train_df["sentiment"])
    y_test = labels_to_int(test_df["sentiment"])

    # build vocabulary only on training data
    vocab = build_vocab(train_tokens, max_vocab_size=max_vocab_size)

    # convert tokens to id sequences
    X_train = np.array([tokens_to_ids(tokens, vocab, seq_len) for tokens in train_tokens], dtype=np.int64)
    X_test = np.array([tokens_to_ids(tokens, vocab, seq_len) for tokens in test_tokens], dtype=np.int64)

    print(f"Total rows: {n_rows}  -> train: {len(train_df)}, test: {len(test_df)}")
    print("Sequence length:", seq_len)
    print("Vocab size:", len(vocab))
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)

    return X_train, y_train, X_test, y_test, vocab


if __name__ == "__main__":
    for L in [25, 50, 100]:
        print("Sequence length:", L)
        X_tr, y_tr, X_te, y_te, vocab = prepare_data(L)
        print()
import torch
import torch.nn as nn

class SentimentModel(nn.Module):
    """
    A simple sentiment classification model using RNN, LSTM, or Bi-LSTM. 
    Each review is passed through an embedding layer, an RNN-based network, and finally a fully connected layer with sigmoid activation to output the probability of the review being positive.
    """

    def __init__(
        self,
        vocab_size,
        rnn_type="lstm",      # "rnn", "lstm", or "bi_lstm"
        embedding_dim=100,     # embedding vector size
        hidden_dim=64,         # hidden layer size
        num_layers=2,          # number of RNN/LSTM layers
        dropout=0.5,           # dropout rate
        activation="relu",     # "relu", "tanh", or "sigmoid"
    ):
        super().__init__()

        # embedding layer converts word IDs into dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # choosing the type of recurrent network
        if rnn_type == "rnn":
            self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout,)
            self.bidirectional = False

        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout,)
            self.bidirectional = False

        elif rnn_type == "bi_lstm":
            self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True,)
            self.bidirectional = True

        else:
            raise ValueError("Invalid rnn_type. Use 'rnn', 'lstm', or 'bi_lstm'")

        # setting the size of the fully connected input depending on direction
        fc_input_dim = hidden_dim * 2 if rnn_type == "bi_lstm" else hidden_dim

        # choosing the activation function for hidden layer
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError("Invalid activation function. Use 'relu', 'tanh', or 'sigmoid'")

        # dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

        # fully connected layer for binary classification (1 output node)
        self.fc = nn.Linear(fc_input_dim, 1)

        # final sigmoid activation gives output between 0 and 1
        self.out = nn.Sigmoid()

    def forward(self, x):
        # step 1: embeddings
        x = self.embedding(x)

        # step 2: pass through RNN/LSTM/Bi_LSTM
        out, _ = self.rnn(x)

        # step 3: take output from the last time step
        out = out[:, -1, :]

        # step 4: dropout + activation
        out = self.dropout(out)
        if self.activation is not None:
            out = self.activation(out)

        # step 5: fully connected + sigmoid
        out = self.fc(out)
        out = self.out(out)  # output in the range [0, 1]
        return out

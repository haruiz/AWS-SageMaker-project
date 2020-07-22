import torch.nn as nn


# class LSTMClassifier(nn.Module):
#     """
#     This is the simple RNN model we will be using to perform Sentiment Analysis.
#     """
#
#     def __init__(self, embedding_dim, hidden_dim, vocab_size, dropout=0.5):
#         """
#         Initialize the model by settingg up the various layers.
#         """
#         super(LSTMClassifier, self).__init__()
#
#         self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2,  dropout=dropout)
#         self.dense1 = nn.Linear(in_features=hidden_dim, out_features=300)
#         self.dense2 = nn.Linear(in_features=300, out_features=100)
#         self.dense3 = nn.Linear(in_features=100, out_features=1)
#         self.dropout1 = nn.Dropout(0.5)
#         self.dropout2 = nn.Dropout(0.2)
#         self.sig = nn.Sigmoid()
#
#         self.word_dict = None
#
#     def forward(self, x):
#         """
#         Perform a forward pass of our model on some input.
#         """
#         x = x.t()
#         lengths = x[0, :]
#         reviews = x[1:, :]
#         embeds = self.embedding(reviews)
#         lstm_out, _ = self.lstm(embeds)
#         out = self.dense1(lstm_out)
#         #out = self.dropout1(out)
#         out = self.dense2(out)
#         #out = self.dropout2(out)
#         out = self.dense3(out)
#         out = out[lengths - 1, range(len(lengths))]
#         return self.sig(out.squeeze())

import torch.nn as nn


class LSTMClassifier(nn.Module):
    """
    This is the simple RNN model we will be using to perform Sentiment Analysis.
    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        """
        Initialize the model by settingg up the various layers.
        """
        super(LSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.dense = nn.Linear(in_features=hidden_dim, out_features=1)
        self.sig = nn.Sigmoid()

        self.word_dict = None

    def forward(self, x):
        """
        Perform a forward pass of our model on some input.
        """
        x = x.t()
        lengths = x[0, :]
        reviews = x[1:, :]
        embeds = self.embedding(reviews)
        lstm_out, _ = self.lstm(embeds)
        out = self.dense(lstm_out)
        out = out[lengths - 1, range(len(lengths))]
        return self.sig(out.squeeze())
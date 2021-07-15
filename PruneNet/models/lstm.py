from nltk.util import pr
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # self.dropout = nn.Dropout(0.2)
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size = hidden_dim, num_layers=num_layers, batch_first=True)
        
    def forward(self, x):
        x = self.embeddings(x)
        # x = self.dropout(x)
        lstm_out, (ht, ct) = self.lstm(x)
        # out = self.linear(out)
        return ht[-1]


import torch
from torch import nn
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """Character-level poetry dataset.

    Each item is one fixed-format poem encoded as token ids. The model learns to
    predict the next character from all previous characters.
    """

    def __init__(self, encoded_poems):
        self.data = torch.tensor(encoded_poems, dtype=torch.long)

    def __getitem__(self, index):
        poem = self.data[index]
        return poem[:-1], poem[1:]

    def __len__(self):
        return len(self.data)


class PoemLstm(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=600,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.loss = nn.CrossEntropyLoss()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x, hidden=None):
        embeds = self.embedding(x)
        output, hidden = self.lstm(embeds, hidden)
        output = self.dropout(output)
        output = self.linear(output)
        output = output.reshape(-1, output.shape[-1])
        return output, hidden


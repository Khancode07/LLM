import torch
import torch.nn as nn


class MiniLLM(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, config.embedding_dim)
        self.position_embedding = nn.Embedding(
            config.context_length, config.embedding_dim
        )

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.embedding_dim,
                nhead=config.num_heads,
                dim_feedforward=config.embedding_dim * 4,
                batch_first=True,
            ),
            num_layers=config.num_layers,
        )

        self.output = nn.Linear(config.embedding_dim, vocab_size)

    def forward(self, x):
        _, length = x.shape
        positions = torch.arange(length, device=x.device)

        embeddings = self.token_embedding(x) + self.position_embedding(positions)

        mask = torch.triu(
            torch.ones(length, length, device=x.device, dtype=torch.bool),
            diagonal=1,
        )

        hidden = self.transformer(embeddings, mask=mask)
        return self.output(hidden)
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [SeqLen, Batch, Dim] or [Batch, SeqLen, Dim]
        # We assume [Batch, SeqLen] input to model, so here we need to add pe
        # But nn.Transformer expects [SeqLen, Batch, Dim] by default unless batch_first=True
        return x + self.pe[:x.size(1), :]

class WAFTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2, max_len=128, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.fc = nn.Linear(d_model, num_classes)
        self.d_model = d_model

    def forward(self, src):
        # src: [Batch, SeqLen]
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x)
        
        # Take the [CLS] token equivalent (encoding of first token) OR average pooling
        # Let's use average pooling for simplicity/robustness
        x_pool = output.mean(dim=1)
        
        logits = self.fc(x_pool)
        return logits

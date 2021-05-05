from .attention_based_2d import Encoder
import torch
MAX_LEN = 28


class ClassificationEncoderDecoderHead(torch.nn.Module):
    def __init__(self, decoder_vocab_size, encoder_dim_input, encoder_dim_internal, encoder_num_layers, max_len=MAX_LEN):
        super().__init__()
        self.encoder = Encoder(encoder_dim_input, encoder_dim_internal, num_layers=encoder_num_layers)
        self.vocab_size = decoder_vocab_size
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.max_len = max_len
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv2d(encoder_dim_internal, self.vocab_size, 1) for _ in range(self.max_len)
        ])

    def forward(self, features, texts=None):
        features = self.encoder(features)
        features = self.pool(features)
        logits = []
        for conv in self.convs:
            logits.append(conv(features))
        logits = torch.stack(logits, dim=1).squeeze_(-1).squeeze_(-1)
        classes = torch.max(logits, dim=2)[1]
        return logits, classes

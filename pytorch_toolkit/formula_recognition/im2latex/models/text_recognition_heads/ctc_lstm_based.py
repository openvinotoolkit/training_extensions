from positional_encodings.positional_encodings import PositionalEncodingPermute2D
import torch


class LSTMEncoderDecoder(torch.nn.Module):
    """ LSTM-based encoder-decoder module. """

    def __init__(self, out_size, encoder_hidden_size=256, encoder_input_size=512, positional_encodings=False):
        super().__init__()
        self.out_size = out_size
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_input_size = encoder_input_size
        self.num_layers = 2
        self.bidirectional = True
        self.num_directions = 2 if self.bidirectional else 1
        self.rnn_encoder = torch.nn.LSTM(self.encoder_input_size, self.encoder_hidden_size,
                                         bidirectional=True, num_layers=self.num_layers,
                                         batch_first=True)
        self.rnn_decoder = torch.nn.LSTM(self.encoder_hidden_size * self.num_directions, self.encoder_hidden_size,
                                         bidirectional=True, num_layers=self.num_layers,
                                         batch_first=True)
        self.fc = torch.nn.Linear(self.encoder_hidden_size * self.num_directions, out_features=self.out_size)
        if positional_encodings:
            self.pe = PositionalEncodingPermute2D(channels=self.encoder_input_size)
        else:
            self.pe = None

    def forward(self, encoded_features, formulas=None):
        if self.pe:
            encoded = self.pe(encoded_features)
            encoded_features = encoded_features + encoded
        encoded_features = encoded_features.permute(0, 2, 3, 1)  # [B, H, W, LSTM_INP_CHANNELS]
        B, H, W, LSTM_INP_CHANNELS = encoded_features.size()
        encoded_features = encoded_features.reshape(B*H, W, LSTM_INP_CHANNELS)

        rnn_out, state = self.rnn_encoder(encoded_features)
        rnn_out, state = self.rnn_decoder(rnn_out, state)  # [B*H, W, LSTM_INP_CHANNELS]
        rnn_out = rnn_out.reshape(B, H, W, self.encoder_hidden_size * self.num_directions)
        rnn_out = rnn_out.mean(dim=1)
        logits = self.fc(rnn_out).permute(1, 0, 2)
        targets = torch.max(logits, dim=2)[1]
        return logits, targets

import torch


class LSTMEncoderDecoder(torch.nn.Module):
    """ LSTM-based encoder-decoder module. """

    def __init__(self, out_size, encoder_hidden_size=256, encoder_input_size=512):
        super().__init__()
        self.out_size = out_size
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_input_size = encoder_input_size
        self.num_layers = 2
        self.rnn_encoder = torch.nn.LSTM(self.encoder_input_size, self.encoder_hidden_size,
                                         bidirectional=True, num_layers=self.num_layers,
                                         batch_first=True)
        self.rnn_decoder = torch.nn.LSTM(self.encoder_input_size, self.encoder_hidden_size,
                                         bidirectional=True, num_layers=self.num_layers,
                                         batch_first=True)
        self.fc = torch.nn.Linear(self.encoder_input_size, out_features=self.out_size)

    def forward(self, encoded_features, formulas=None):
        encoded_features = encoded_features.permute(0, 2, 3, 1)  # [B, H, W, LSTM_INP_CHANNELS]
        B, H, W, LSTM_INP_CHANNELS = encoded_features.size()
        encoded_features = encoded_features.reshape(B*H, W, LSTM_INP_CHANNELS)

        rnn_out, state = self.rnn_encoder(encoded_features)
        rnn_out, state = self.rnn_decoder(rnn_out, state)  # [B*H, W, LSTM_INP_CHANNELS]
        rnn_out = rnn_out.reshape(B, H, W, LSTM_INP_CHANNELS)
        rnn_out = rnn_out.mean(dim=[1])
        logits = torch.nn.functional.softmax(self.fc(rnn_out), dim=2)
        targets = torch.max(logits, dim=2)[1]  # [B*H, W]
        return logits, targets

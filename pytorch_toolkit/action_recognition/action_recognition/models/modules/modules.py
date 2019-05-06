import torch
from torch import nn

from .functional import squash_dims


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class Identity(nn.Module):
    def forward(self, input_):
        return input_


class StateInitFC(nn.Module):
    def __init__(self, init_size, hidden_size, activation=Identity):
        super().__init__()

        self.linear_h = nn.Linear(init_size, hidden_size)
        self.linear_c = nn.Linear(init_size, hidden_size)
        self.activation_h = activation()
        self.activation_c = activation()

        self.linear_h.weight.data.normal_(0.0, 0.02)
        self.linear_h.bias.data.fill_(0)
        self.linear_c.weight.data.normal_(0.0, 0.02)
        self.linear_c.bias.data.fill_(0)

    def forward(self, input_):
        h0 = self.activation_h(self.linear_h(input_))
        c0 = self.activation_c(self.linear_c(input_))
        return h0, c0


class StateInitZero(nn.Module):
    def __init__(self, hidden_size, num_layers=1, batch_first=False):
        super(StateInitZero, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

    def forward(self, input: torch.Tensor):
        h0 = input.new_zeros((self.num_layers, input.size(0 if self.batch_first else 1), self.hidden_size))
        c0 = input.new_zeros((self.num_layers, input.size(0 if self.batch_first else 1), self.hidden_size))
        return h0, c0


class Attention(nn.Module):
    def __init__(self, q_size, k_size, v_size):
        super().__init__()

        self.softmax = nn.Softmax(dim=1)
        self.linear_q = nn.Linear(q_size, v_size)

        self.linear_q.weight.data.normal_(0.0, 0.02)
        self.linear_q.bias.data.fill_(0)

    def forward(self, q, k, v):
        attn_scores = self.linear_q(q)
        attn_map = self.softmax(attn_scores.view(-1, attn_scores.size(-1)))

        return (v * attn_map).sum(-1), attn_map


class AttentionLSTM(nn.Module):
    """LSTM with spatial attention """

    def __init__(self, input_features, hidden_size, attention_size, batch_first=False, **kwargs):
        super().__init__()

        self.batch_first = batch_first
        self.lstm = nn.LSTM(input_features, hidden_size, batch_first=False, **kwargs)
        self.attention = Attention(hidden_size, None, attention_size)

    def forward(self, x, hidden):
        hx, cx = hidden
        if self.batch_first:
            x = x.transpose(0, 1)

        outputs = []
        for i in range(x.size(0)):
            # transpose in order to correctly broadcast while multiplying v
            # squash dims in order to pull into vector (C x N x L)
            v = squash_dims(x[i].transpose(0, 1), (2, 3))
            feature, attention = self.attention(hx[-1], v, v)
            feature = feature.transpose(0, 1)  # back to (N x C)

            # unsqueeze to emulate sequence size = 1
            _, (hx, cx) = self.lstm(feature.unsqueeze(0), (hx, cx))
            outputs.append(hx)
        ys = torch.cat(outputs, 0)

        if self.batch_first:
            ys = ys.transpose(0, 1)

        return ys, (hx, cx)

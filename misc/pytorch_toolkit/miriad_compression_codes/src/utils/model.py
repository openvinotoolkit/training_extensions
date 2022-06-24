import torch
from torch import nn


def load_checkpoint(model, checkpoint):
    if checkpoint is not None:
        model_checkpoint = torch.load(checkpoint)
        model.load_state_dict(model_checkpoint['state_dict'])
    else:
        model.state_dict()


class Encoder(nn.Module):
    def __init__(self, n_downconv=3, n_encowidth=64):
        super().__init__()
        # a tunable number of DownConv blocks in the architecture
        self.n_downconv = n_downconv
        self.n_encowidth = n_encowidth
        # The two mandatory initial layers
        layer_list = [
            nn.Conv2d(in_channels=1, out_channels=self.n_encowidth,
                      kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=self.n_encowidth, out_channels=self.n_encowidth,
                      kernel_size=3, stride=2, padding=1), nn.ReLU()
        ]
        for _ in range(self.n_downconv):
            layer_list.extend([
                nn.Conv2d(in_channels=self.n_encowidth, out_channels=self.n_encowidth,
                          kernel_size=3, stride=1, padding=1), nn.ReLU(),
                nn.Conv2d(in_channels=self.n_encowidth, out_channels=self.n_encowidth,
                          kernel_size=3, stride=2, padding=1), nn.ReLU(),
            ])
        # The one mandatory end layer
        layer_list.append(
            nn.Conv2d(in_channels=self.n_encowidth, out_channels=16,
                      kernel_size=3, stride=1, padding=1)
        )
        # register the Sequential module
        self.encoder = nn.Sequential(*layer_list)

    def forward(self, x):
        # forward pass; a final clamping is applied
        return torch.clamp(self.encoder(x), 0, 1)


class Decoder(nn.Module):
    def __init__(self, n_upconv=3, n_decowidth=64):
        super().__init__()

        # a tunable number of DownConv blocks in the architecture
        self.n_upconv = n_upconv
        self.n_decowidth = n_decowidth

        # The one mandatory initial layers
        layer_list = [
            nn.Conv2d(in_channels=16, out_channels=n_decowidth,
                      kernel_size=3, stride=1, padding=1), nn.ReLU(),
        ]
        # 'n_upconv' number of UpConv layers (In the CVPR paper, it was 3)
        for _ in range(self.n_upconv):
            layer_list.extend([
                nn.Conv2d(in_channels=n_decowidth, out_channels=n_decowidth *
                          4, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                nn.PixelShuffle(2)
            ])
        # The mandatory final layer
        layer_list.extend([
            nn.Conv2d(in_channels=n_decowidth, out_channels=1 *
                      4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2)
        ])
        # register the Sequential module
        self.decoder = nn.Sequential(*layer_list)

    def forward(self, x):
        # forward pass; a final clamping is applied
        return torch.clamp(self.decoder(x), 0, 1)


class AutoEncoder(nn.Module):
    def __init__(self, n_updownconv=3, width=64):
        super().__init__()
        self.n_updownconv = n_updownconv
        self.width = width

        # there must be same number of 'n_downconv' and 'n_upconv'
        self.encoder = Encoder(
            n_downconv=self.n_updownconv, n_encowidth=self.width)
        self.decoder = Decoder(
            n_upconv=self.n_updownconv, n_decowidth=self.width)

    def forward(self, x):
        self.shape_input = list(x.shape)
        x = self.encoder(x)
        self.shape_latent = list(x.shape)
        x = self.decoder(x)
        return x

import torch
from torch import nn
import torchvision

from modules.conv import conv


class InitialStage(nn.Module):
    def __init__(self, num_channels, num_heatmaps, num_pafs):
        super().__init__()
        self.heatmaps_features = nn.Sequential(
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False)
        )
        self.pafs_features = nn.Sequential(
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False)
        )
        self.heatmaps = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        heatmaps_features = self.heatmaps_features(x)
        pafs_features = self.pafs_features(x)
        heatmaps = self.heatmaps(heatmaps_features)
        pafs = self.pafs(pafs_features)
        return [heatmaps, pafs]


class RefinementStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_heatmaps, num_pafs):
        super().__init__()
        self.heatmaps_features = nn.Sequential(
            conv(in_channels, out_channels, kernel_size=7, padding=3, bn=False),
            conv(out_channels, out_channels, kernel_size=7, padding=3, bn=False),
            conv(out_channels, out_channels, kernel_size=7, padding=3, bn=False),
            conv(out_channels, out_channels, kernel_size=7, padding=3, bn=False),
            conv(out_channels, out_channels, kernel_size=7, padding=3, bn=False)
        )
        self.pafs_features = nn.Sequential(
            conv(in_channels, out_channels, kernel_size=7, padding=3, bn=False),
            conv(out_channels, out_channels, kernel_size=7, padding=3, bn=False),
            conv(out_channels, out_channels, kernel_size=7, padding=3, bn=False),
            conv(out_channels, out_channels, kernel_size=7, padding=3, bn=False),
            conv(out_channels, out_channels, kernel_size=7, padding=3, bn=False)
        )
        self.heatmaps = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        heatmaps_features = self.heatmaps_features(x)
        pafs_features = self.pafs_features(x)
        heatmaps = self.heatmaps(heatmaps_features)
        pafs = self.pafs(pafs_features)
        return [heatmaps, pafs]


class PoseEstimationWithVgg19(nn.Module):
    def __init__(self, num_refinement_stages=1, num_channels=128, num_heatmaps=19, num_pafs=38):
        super().__init__()
        vgg19 = torchvision.models.vgg19(pretrained=True)
        self.model = vgg19.features[0:23]
        self.cpm = nn.Sequential(
            conv(512, 256, bn=False),           # conv4_3_cpm
            conv(256, num_channels, bn=False)   # conv4_4_cpm
        )

        self.initial_stage = InitialStage(num_channels, num_heatmaps, num_pafs)
        self.refinement_stages = nn.ModuleList()
        for idx in range(num_refinement_stages):
            self.refinement_stages.append(RefinementStage(num_channels + num_heatmaps + num_pafs, num_channels,
                                                          num_heatmaps, num_pafs))

    def forward(self, x):
        backbone_features = self.model(x)
        backbone_features = self.cpm(backbone_features)

        stages_output = self.initial_stage(backbone_features)
        for refinement_stage in self.refinement_stages:
            stages_output.extend(
                refinement_stage(torch.cat([backbone_features, stages_output[-2], stages_output[-1]], dim=1)))

        return stages_output

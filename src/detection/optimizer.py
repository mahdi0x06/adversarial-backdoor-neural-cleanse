import torch
import torch.nn as nn


class TriggerOptimizer(nn.Module):

    def __init__(
        self,
        channels=3,
        height=32,
        width=32,
    ):
        super().__init__()

        self.mask_param = nn.Parameter(
            torch.zeros(
                1,
                1,
                height,
                width,
            )
        )

        self.pattern_param = nn.Parameter(
            torch.zeros(
                1,
                channels,
                height,
                width,
            )
        )

    def get_mask(self):

        return torch.sigmoid(
            self.mask_param
        )

    def get_pattern(self):

        return torch.sigmoid(
            self.pattern_param
        )

    def forward(self, images):

        mask = self.get_mask()

        pattern = self.get_pattern()

        poisoned = (
            (1.0 - mask) * images
            +
            mask * pattern
        )

        return (
            poisoned,
            mask,
            pattern,
        )
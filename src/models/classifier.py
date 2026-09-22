import torch
import torch.nn as nn


CIFAR10_MEAN = (
    0.4914,
    0.4822,
    0.4465,
)

CIFAR10_STD = (
    0.2470,
    0.2435,
    0.2616,
)


class Normalize(nn.Module):

    def __init__(
        self,
        mean=CIFAR10_MEAN,
        std=CIFAR10_STD,
    ):
        super().__init__()

        mean = torch.tensor(
            mean,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            std,
            dtype=torch.float32,
        ).view(1, 3, 1, 1)

        self.register_buffer(
            "mean",
            mean,
        )

        self.register_buffer(
            "std",
            std,
        )

    def forward(self, x):
        return (x - self.mean) / self.std


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(
            out_channels
        )

        self.relu = nn.ReLU(
            inplace=True
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(
            out_channels
        )

        if (
            stride != 1
            or in_channels != out_channels
        ):
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    out_channels
                ),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out


class ResNet20(nn.Module):

    def __init__(
        self,
        num_classes: int = 10,
    ):
        super().__init__()

        self.normalize = Normalize()

        self.in_channels = 16

        self.conv1 = nn.Conv2d(
            3,
            16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(
            16
        )

        self.relu = nn.ReLU(
            inplace=True
        )

        self.layer1 = self._make_layer(
            out_channels=16,
            num_blocks=3,
            stride=1,
        )

        self.layer2 = self._make_layer(
            out_channels=32,
            num_blocks=3,
            stride=2,
        )

        self.layer3 = self._make_layer(
            out_channels=64,
            num_blocks=3,
            stride=2,
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.fc = nn.Linear(
            64,
            num_classes,
        )

        self._initialize_weights()

    def _make_layer(
        self,
        out_channels: int,
        num_blocks: int,
        stride: int,
    ):
        strides = [
            stride
        ] + [
            1
        ] * (
            num_blocks - 1
        )

        blocks = []

        for block_stride in strides:
            blocks.append(
                BasicBlock(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=block_stride,
                )
            )

            self.in_channels = out_channels

        return nn.Sequential(
            *blocks
        )

    def _initialize_weights(self):
        for module in self.modules():

            if isinstance(
                module,
                nn.Conv2d,
            ):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

            elif isinstance(
                module,
                nn.BatchNorm2d,
            ):
                nn.init.constant_(
                    module.weight,
                    1,
                )

                nn.init.constant_(
                    module.bias,
                    0,
                )

    def forward(self, x):
        x = self.normalize(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avg_pool(x)

        x = torch.flatten(
            x,
            1,
        )

        x = self.fc(x)

        return x


def create_classifier(
    num_classes: int = 10,
) -> ResNet20:

    return ResNet20(
        num_classes=num_classes,
    )
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        f1: int,
        f3_reduce: int,
        f3: int,
        f5_reduce: int,
        f5: int,
        pool_proj: int,
    ) -> None:
        super().__init__()

        self.proj1 = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        self.proj2 = nn.Sequential(
            nn.Conv2d(in_channels, f3_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f3_reduce, f3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.proj3 = nn.Sequential(
            nn.Conv2d(in_channels, f5_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f5_reduce, f5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )

        self.proj4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p1 = self.proj1(x)
        p2 = self.proj2(x)
        p3 = self.proj3(x)
        p4 = self.proj4(x)
        return torch.cat([p1, p2, p3, p4], dim=1)

    @property
    def out_channels(self) -> int:
        return (
            self.proj1[0].out_channels
            + self.proj2[2].out_channels
            + self.proj3[2].out_channels
            + self.proj4[1].out_channels
        )


class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Linear(128 * 2 * 2, 1024)
        self.dropout = nn.Dropout(p=0.7)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.dropout(x)
        return self.fc2(x)


class InceptionNet(nn.Module):
    _BLOCK_CONFIGS: list[tuple[int, int, int, int, int, int]] = [
        (64, 96, 128, 16, 32, 32),
        (128, 128, 192, 32, 96, 64),
        (192, 96, 208, 16, 48, 64),
        (160, 112, 224, 24, 64, 64),
        (128, 128, 256, 24, 64, 64),
        (112, 144, 288, 32, 64, 64),
        (256, 160, 320, 32, 128, 128),
        (256, 160, 320, 32, 128, 128),
        (384, 192, 384, 48, 128, 128),
    ]

    _IN_CHANNELS: list[int] = [192, 256, 480, 512, 512, 512, 528, 832, 832]

    def __init__(
        self,
        num_classes: int = 10,
        aux_logits: bool = True,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.aux_logits = aux_logits

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        cfgs = self._BLOCK_CONFIGS
        ins = self._IN_CHANNELS

        self.inception3a = InceptionBlock(ins[0], *cfgs[0])
        self.inception3b = InceptionBlock(ins[1], *cfgs[1])
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionBlock(ins[2], *cfgs[2])
        self.inception4b = InceptionBlock(ins[3], *cfgs[3])
        self.inception4c = InceptionBlock(ins[4], *cfgs[4])
        self.inception4d = InceptionBlock(ins[5], *cfgs[5])
        self.inception4e = InceptionBlock(ins[6], *cfgs[6])
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = InceptionBlock(ins[7], *cfgs[7])
        self.inception5b = InceptionBlock(ins[8], *cfgs[8])

        if aux_logits:
            self.aux1 = AuxiliaryClassifier(512, num_classes)
            self.aux2 = AuxiliaryClassifier(528, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(1024, num_classes)

        self._initialize_weights()

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv3(self.conv2(x)))

        x = self.pool3(self.inception3b(self.inception3a(x)))

        x = self.inception4a(x)
        aux1_out = self.aux1(x) if (self.aux_logits and self.training) else None

        x = self.inception4d(self.inception4c(self.inception4b(x)))
        aux2_out = self.aux2(x) if (self.aux_logits and self.training) else None

        x = self.pool4(self.inception4e(x))

        x = self.inception5b(self.inception5a(x))

        x = self.avgpool(x)
        x = self.fc(self.dropout(x.view(x.size(0), -1)))

        if self.training and self.aux_logits:
            return x, aux1_out, aux2_out
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class InceptionNetLoss(nn.Module):
    def __init__(self, aux_weight: float = 0.3) -> None:
        super().__init__()
        self.aux_weight = aux_weight
        self._ce = nn.CrossEntropyLoss()

    def forward(
        self,
        outputs: torch.Tensor | tuple[torch.Tensor, ...],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(outputs, torch.Tensor):
            return self._ce(outputs, targets)

        main_logits, aux1_logits, aux2_logits = outputs
        loss = self._ce(main_logits, targets)

        if aux1_logits is not None:
            loss = loss + self.aux_weight * self._ce(aux1_logits, targets)
        if aux2_logits is not None:
            loss = loss + self.aux_weight * self._ce(aux2_logits, targets)

        return loss
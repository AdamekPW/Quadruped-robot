
import torch.nn as nn


class ConvBlock(nn.Module):
    """Blok konwolucyjny: Conv → GroupNorm → ReLU (+ opcjonalny MaxPool).

    Używamy GroupNorm, a NIE BatchNorm: normalizuje per-próbkę (bez statystyk
    batcha, identycznie w train/eval), więc jest bezpieczna w RL on-policy i nie
    pozwala aktywacjom się zapaść. Bez żadnej normalizacji gałąź CNN wypadała w
    martwy stan (wyjście ~0, znikomy gradient) i polityka była ślepa na teren.
    Przesunięcie GroupNorm (beta) zastępuje bias, więc conv może mieć bias=False.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        bias=False,
        use_pooling=False,
        num_groups=8,
    ) -> None:
        super().__init__()

        # Liczba grup musi dzielić liczbę kanałów.
        groups = min(num_groups, out_channels)
        while out_channels % groups != 0:
            groups -= 1

        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            ),
            nn.GroupNorm(num_groups=groups, num_channels=out_channels),
            nn.ReLU(inplace=True),
        ]

        if use_pooling:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

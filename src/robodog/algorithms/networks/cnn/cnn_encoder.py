"""Wspólna baza sieci z enkoderem konwolucyjnym nad mapą terenu.

Aktor i krytyk różnią się tylko tym, którą grupę proprioceptywną dostają
(`obs_set`), jaki mają wymiar wyjścia (`output_dim`) i czy mają rozkład akcji
(`distribution_cfg`) — a to wszystko przekazuje PPO przy tworzeniu modelu. Cała
logika „widzenia" (CNN nad `height_scan` + sklejenie z propriocepcją) jest
wspólna i mieszka tutaj.
"""

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP, HiddenState
from tensordict import TensorDict

from robodog.algorithms.networks.cnn.conv_block import ConvBlock


class CNNEncoderModel(MLPModel):
    """MLPModel z enkoderem CNN dla obserwacji-obrazu (mapa wysokości terenu).

    Grupa obserwacji `image_group` (kształt `(B, C, H, W)`) przechodzi przez trzy
    bloki konwolucyjne, spłaszcza się do wektora cech i skleja z (znormalizowaną)
    propriocepcją. Głowę `self.mlp`, rozkład akcji i próbkowanie dziedziczymy z
    `MLPModel` — nadpisujemy tylko `get_latent` (czyli „jak budujemy cechy").
    """

    # Nazwa grupy obserwacji będącej obrazem terenu (poza `obs_groups` — MLPModel
    # nie przyjmuje grup 2D, więc czytamy ją wprost z `obs`).
    image_group: str = "height_scan"

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        cnn_cfg=None,
        **kwargs,
    ):
        # Baza buduje: grupy proprioceptywne + normalizer + rozkład + domyślny mlp.
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )

        # Enkoder terenu: liczbę kanałów bierzemy z realnego kształtu obrazu.
        image = obs[self.image_group]
        in_channels = image.shape[1]
        self.conv1_block = ConvBlock(in_channels, 16, use_pooling=True)
        self.conv2_block = ConvBlock(16, 32, use_pooling=True)
        self.conv3_block = ConvBlock(32, 64, use_pooling=True)

        # Wymiar cech po konwolucjach liczymy dynamicznie (próbny przebieg),
        # żeby nie zaszywać go na sztywno — przy zmianie siatki terenu sam się
        # dopasuje.
        cnn_dim = self._cnn_feature_dim(image.shape[1:])

        # Głowa: wejście = cechy CNN + propriocepcja (self.obs_dim z bazy).
        # Wyjście = input_dim rozkładu (aktor) albo output_dim (krytyk bez rozkładu).
        out_dim = (
            self.distribution.input_dim if self.distribution is not None else output_dim
        )
        self.mlp = MLP(cnn_dim + self.obs_dim, out_dim, hidden_dims, activation)
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

    def _encode_terrain(self, image: torch.Tensor) -> torch.Tensor:
        """Przepuszcza mapę terenu przez bloki konwolucyjne i spłaszcza do wektora."""
        z = self.conv1_block(image)
        z = self.conv2_block(z)
        z = self.conv3_block(z)
        return torch.flatten(z, start_dim=1)

    @torch.no_grad()
    def _cnn_feature_dim(self, image_shape: torch.Size) -> int:
        """Liczba cech po spłaszczeniu konwolucji — z próbnego przebiegu (B=1)."""
        dummy = torch.zeros(1, *image_shape)
        return self._encode_terrain(dummy).shape[1]

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        proprio = super().get_latent(obs, masks, hidden_state)
        terrain = self._encode_terrain(obs[self.image_group])
        return torch.cat((proprio, terrain), dim=1)

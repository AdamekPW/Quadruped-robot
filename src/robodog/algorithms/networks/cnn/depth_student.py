""" 
Student do distylacji (imitation/DAgger).

"""

from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP, RNN, HiddenState
from tensordict import TensorDict
import torch

from robodog.algorithms.networks.cnn.conv_block import ConvBlock


class DepthStudent(MLPModel):
    """ Sieć studenta """

    # Grupa-obraz to GŁĘBIA - jedyne wejście eksteroceptywne dostępne na sprzęcie.
    image_group = "depth"

    is_recurrent = True

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
        rnn_type: str = "lstm",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
    ):
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

        image = obs[self.image_group]
        in_channels = image.shape[1]
        self.conv1_block = ConvBlock(in_channels, 16, use_pooling=True)
        self.conv2_block = ConvBlock(16, 32, use_pooling=True)
        self.conv3_block = ConvBlock(32, 64, use_pooling=True)

        cnn_dim = self._cnn_feature_dim(image.shape[1:])

        out_dim = (
            self.distribution.input_dim if self.distribution is not None else output_dim
        )

        self.memory_block = RNN(
            input_size=cnn_dim + self.obs_dim,
            hidden_dim=rnn_hidden_dim,
            num_layers=rnn_num_layers,
            type=rnn_type,
        )

        self.mlp = MLP(rnn_hidden_dim, out_dim, hidden_dims, activation)
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)


    def _encode_terrain(self, image: torch.Tensor) -> torch.Tensor:
        """Przepuszcza mapę terenu przez bloki konwolucyjne i spłaszcza do wektora."""
        z = self.conv1_block(image)
        z = self.conv2_block(z)
        z = self.conv3_block(z)
        return torch.flatten(z, start_dim=1)

    def reset(self, dones=None, hidden_state=None):
        """Czyści pamięć robota"""
        self.memory_block.reset(dones, hidden_state)

    @torch.no_grad()
    def _cnn_feature_dim(self, image_shape: torch.Size) -> int:
        """Liczba cech po spłaszczeniu konwolucji - z próbnego przebiegu (B=1)."""
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

        latent_raw = torch.cat((proprio, terrain), dim=1)

        return self.memory_block(latent_raw, masks, hidden_state).squeeze(0)

    def get_hidden_state(self) -> HiddenState:
        return self.memory_block.hidden_state # type: ignore

    def detach_hidden_state(self, dones=None):
        self.memory_block.detach_hidden_state(dones)

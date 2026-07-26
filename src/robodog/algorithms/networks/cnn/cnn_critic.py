"""Krytyk z enkoderem CNN nad mapą terenu.

Cała logika jest wspólna z aktorem i mieszka w `CNNEncoderModel`. Krytyka
odróżnia to, co przekazuje PPO: grupa `critic` (obserwacje uprzywilejowane),
`output_dim=1` (wartość stanu) oraz brak `distribution_cfg` (krytyk nie próbkuje).
"""

from robodog.algorithms.networks.cnn.cnn_encoder import CNNEncoderModel


class CNN_critic(CNNEncoderModel):
    """Krytyk: propriocepcja `obs["critic"]` + CNN nad terenem → wartość stanu."""

    pass

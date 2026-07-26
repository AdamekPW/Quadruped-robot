"""Aktor z enkoderem CNN nad mapą terenu.

Cała logika jest wspólna z krytykiem i mieszka w `CNNEncoderModel`. Aktora od
krytyka odróżnia tylko to, co przekazuje PPO przy tworzeniu (grupa `actor`,
`output_dim` = liczba akcji, oraz `distribution_cfg` — rozkład akcji).
"""

from robodog.algorithms.networks.cnn.cnn_encoder import CNNEncoderModel


class CNN_actor(CNNEncoderModel):
    """Aktor: propriocepcja `obs["actor"]` + CNN nad terenem → średnie akcji."""

    pass

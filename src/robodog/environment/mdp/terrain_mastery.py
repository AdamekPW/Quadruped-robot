"""
Pomiar stopnia opanowania terenu — ile procent siatki robot faktycznie przechodzi.

Po co, skoro jest już `Curriculum/terrain_levels`? Bo średni poziom ma sufit.
Robot, który zaliczy najwyższy wiersz, jest przez mjlab przerzucany na LOSOWY
poziom (`TerrainEntity.update_env_origins`), więc nawet polityka wygrywająca
zawsze utknie na średniej 2*(num_rows-1)/3, czyli 6.0 przy dziesięciu wierszach.
Metryka nie odróżnia wtedy polityki doskonałej od takiej, która przegrywa co
piąty epizod (5.72). Tutaj mierzymy co innego: dla KAŻDEJ komórki siatki
(poziom x typ terenu) odsetek epizodów zakończonych przejściem wymaganego
dystansu. 100% znaczy "wygrywa wszędzie, łącznie z najtrudniejszym wierszem".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.curriculum_manager import CurriculumTermCfg


class terrain_mastery:
    """Odsetek zaliczonych epizodów per komórka siatki terenu.

    Człon curriculum, który NICZEGO nie zmienia — tylko mierzy. Siedzi tutaj, a nie
    wśród metryk, bo człony metryk liczą wartość co krok dla każdego środowiska,
    a ten pomiar ma sens raz na epizod, przy resecie, i musi wiedzieć, KTÓRE
    środowiska się resetują.

    KOLEJNOŚĆ W SŁOWNIKU `curriculum` MA ZNACZENIE. Bazowy człon `terrain_levels`
    przy okazji awansu nadpisuje `terrain_levels`, `terrain_types` i `env_origins`
    resetowanych środowisk. Puszczony po nim ten człon liczyłby dystans względem
    nowego originu i przypisywał wynik do nowej komórki — po cichu, bez błędu.
    Dlatego musi być PIERWSZY.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv) -> None:
        del cfg  # Parametry przychodzą przy każdym wywołaniu.

        terrain = env.scene.terrain
        if terrain is None or terrain.terrain_origins is None:
            raise ValueError(
                "terrain_mastery wymaga terenu z generatora "
                "(terrain_type='generator'); płaski `plane` nie ma siatki do pomiaru."
            )
        generator = terrain.cfg.terrain_generator
        if generator is None:
            raise ValueError("terrain_mastery: teren nie ma konfiguracji generatora.")

        self._num_rows, self._num_cols = terrain.terrain_origins.shape[:2]

        # Sukces = ten sam próg, który napędza awans w `terrain_levels_vel`. Celowo,
        # żeby obie metryki mówiły o tym samym zdarzeniu.
        self._success_distance = generator.size[0] / 2

        # W trybie curriculum generator robi jedną kolumnę na typ terenu, więc indeks
        # kolumny to wprost nazwa sub-terenu. Gdyby to przestało być prawdą, lecimy
        # na indeksach zamiast zgadywać.
        names = list(generator.sub_terrains.keys())
        self._column_names = (
            names
            if len(names) == self._num_cols
            else [f"col_{i}" for i in range(self._num_cols)]
        )

        # Stan: średnia krocząca sukcesu dla każdej komórki siatki. Komórki jeszcze
        # nieodwiedzone zostają na zerze, więc metryka rośnie od zera zamiast
        # udawać sukces tam, gdzie robot jeszcze nie był.
        self._success_rate = torch.zeros(
            self._num_rows * self._num_cols, device=env.device
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
        smoothing: float = 0.01,
    ) -> dict[str, torch.Tensor]:
        if not 0.0 < smoothing <= 1.0:
            raise ValueError(
                f"terrain_mastery: smoothing musi być w (0, 1], dostano {smoothing}."
            )

        terrain = env.scene.terrain
        assert terrain is not None

        asset = env.scene["robot"]
        distance = torch.norm(
            asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
            dim=1,
        )
        success = (distance > self._success_distance).float()

        # Komórka siatki spłaszczona do jednego indeksu, żeby zliczyć wszystko
        # jednym `bincount` zamiast pętli po 70 komórkach.
        cells = (
            terrain.terrain_levels[env_ids] * self._num_cols
            + terrain.terrain_types[env_ids]
        ).long()
        num_cells = self._success_rate.numel()
        counts = torch.bincount(cells, minlength=num_cells).float()
        successes = torch.bincount(cells, weights=success, minlength=num_cells)

        # Wygładzanie liczone OD PRÓBKI, nie od wywołania: komórka, która dostała
        # w tej paczce pięć epizodów, przesuwa się tak, jakby EMA zadziałała pięć
        # razy. Bez tego rzadko odwiedzane wiersze (a rozkład wizyt rośnie liniowo
        # z poziomem) uczyłyby się tak samo szybko jak te tłoczne.
        visited = counts > 0
        weight = 1.0 - (1.0 - smoothing) ** counts[visited]
        batch_rate = successes[visited] / counts[visited]
        self._success_rate[visited] = (
            self._success_rate[visited] * (1.0 - weight) + batch_rate * weight
        )

        # Raportujemy w procentach — metryka ma się czytać wprost z wykresu.
        grid = self._success_rate.view(self._num_rows, self._num_cols) * 100.0
        result: dict[str, torch.Tensor] = {"mean": grid.mean()}
        for column, name in enumerate(self._column_names):
            result[name] = grid[:, column].mean()
        for row in range(self._num_rows):
            result[f"level_{row}"] = grid[row].mean()
        return result

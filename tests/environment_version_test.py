"""Testy dymne (smoke tests) środowiska.

Nie sprawdzają logiki projektu — sprawdzają, czy środowisko w ogóle wstało
poprawnie. Sens: po przeniesieniu projektu na inną maszynę (np. klaster
uczelniany) jedno `pytest` odpowiada na pytanie "czy stack działa", zanim
zmarnujesz godziny na debugowanie treningu, który nie miał prawa ruszyć.
"""

from importlib.metadata import version

import pytest


def test_robodog_package_is_importable():
    """Wymaga `pip install -e .` — łapie najczęstszy błąd konfiguracji."""
    import robodog

    assert robodog.__doc__ is not None


def test_key_dependency_versions():
    """Piny z environment.yml zgadzają się z tym, co faktycznie zainstalowano."""
    import mujoco
    import torch

    assert torch.__version__.startswith("2.11.0")
    assert mujoco.__version__.startswith("3.10.0")


def test_torch_sees_gpu():
    """Bez CUDY trening PPO będzie liczony na CPU — czyli boleśnie wolno."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("brak GPU — dozwolone na maszynie bez karty NVIDIA")

    assert torch.cuda.device_count() >= 1
    # Realny rachunek na GPU: samo `is_available()` bywa prawdziwe nawet wtedy,
    # gdy kernele dla danej architektury nie są w buildzie (np. Blackwell sm_120
    # na buildzie starszym niż cu128) i wysypują się dopiero przy obliczeniu.
    x = torch.randn(8, 8, device="cuda")
    assert torch.matmul(x, x).isfinite().all().item()


def test_mujoco_warp_has_no_known_regression():
    """mujoco-warp MUSI być 3.10.0.1 — 3.10.0.2 ma regresję blokującą trening.

    W 3.10.0.2 zadania `velocity` (Go1 i G1) padają z CUDA error 700 (illegal
    memory access) przy num_envs >= ~176; przy 160 jeszcze działa, a cartpole
    (bez randomizacji fizyki) działa nawet przy 4096. Ścieżka awarii:
    zdarzenia startowe randomizacji -> sim.recompute_constants() -> smooth.crb().

    To blokuje projekt, bo trening potrzebuje 4096 środowisk.

    mjlab pinuje `mujoco-warp>=3.10.0.2`, więc pip sam zainstaluje zepsutą
    wersję, a poprawnej nie da się zapisać w environment.yml (pip zgłasza
    ResolutionImpossible). Ten test jest jedynym zabezpieczeniem — szczegóły
    w docs/mjlab-architektura.md.
    """
    assert version("mujoco-warp") == "3.10.0.1", (
        "Wykryto mujoco-warp != 3.10.0.1. Jeśli to 3.10.0.2, trening padnie przy "
        "num_envs >= ~176. Napraw: pip install --no-deps mujoco-warp==3.10.0.1"
    )


def test_mjlab_quadruped_task_is_registered():
    """Zadanie chodzenia czworonoga musi być widoczne w rejestrze mjlab.

    Go1, nie Go2 — mjlab nie dostarcza Go2 (patrz docs/mjlab-architektura.md).
    Ten test trzeba będzie rozszerzyć o Go2 po wykonaniu portu.
    """
    from mjlab.tasks.registry import list_tasks

    assert "Mjlab-Velocity-Flat-Unitree-Go1" in list_tasks()

"""Testy dymne (smoke tests) środowiska.

Nie sprawdzają logiki projektu — sprawdzają, czy środowisko w ogóle wstało
poprawnie. Sens: po przeniesieniu projektu na inną maszynę (np. klaster
uczelniany) jedno `pytest` odpowiada na pytanie "czy stack działa", zanim
zmarnujesz godziny na debugowanie treningu, który nie miał prawa ruszyć.
"""
import pytest


def test_robodog_package_is_importable():
    """Wymaga `pip install -e .` — łapie najczęstszy błąd konfiguracji."""
    import robodog

    assert robodog.__doc__ is not None


def test_key_dependency_versions():
    """Piny z environment.yml zgadzają się z tym, co faktycznie zainstalowano."""
    import gymnasium
    import mujoco
    import torch

    assert torch.__version__.startswith("2.11.0")
    assert mujoco.__version__.startswith("3.10.0")
    assert gymnasium.__version__.startswith("1.3.0")


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


def test_mujoco_environment_works():
    """Ant-v5 = czworonóg z Gymnasium, tymczasowy zamiennik robopsa."""
    import gymnasium as gym

    env = gym.make("Ant-v5")
    try:
        obs, _ = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape

        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        assert obs.shape == env.observation_space.shape
        assert isinstance(float(reward), float)
    finally:
        env.close()

# RoboDog 🐕‍🦺

Praca magisterska: **nauka chodzenia robopsa metodą uczenia przez wzmacnianie (Reinforcement Learning)**.

## Stack technologiczny

- **Python 3.12**
- **PyTorch (CUDA / cu128)** — sieci neuronowe i trening (GPU: NVIDIA RTX 3060)
- **MuJoCo** — symulator fizyki
- **Gymnasium** — API środowisk RL
- **Algorytm:** własna implementacja **PPO** w PyTorch

## Struktura projektu

```
RoboDog/
├── configs/          # Konfiguracje eksperymentów (YAML): hiperparametry, ustawienia env
├── assets/           # Model robopsa dla MuJoCo
│   ├── robodog.xml   #   Definicja MJCF (geometria, stawy, siłowniki)
│   └── meshes/       #   Siatki 3D i tekstury
├── src/robodog/      # Główny pakiet (instalowalny: pip install -e .)
│   ├── envs/         #   Środowiska Gymnasium owijające MuJoCo (obserwacje, akcje, nagroda)
│   ├── algorithms/   #   Własne implementacje RL (PPO) + sieci actor/critic
│   └── utils/        #   Wczytywanie konfiguracji, logowanie, seedy
├── scripts/          # Punkty wejścia CLI: train.py, evaluate.py, record_video.py
├── notebooks/        # Analiza wyników, wykresy krzywych uczenia do pracy
├── tests/            # Testy środowiska (kształty obserwacji, granice akcji)
├── runs/             # Logi / TensorBoard (ignorowane w git)
├── checkpoints/      # Zapisane modele (ignorowane w git)
└── requirements.txt  # Zależności (PyTorch z indeksu CUDA)
```

## Instalacja

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Status

🚧 Projekt w fazie inicjalizacji — struktura szkieletowa. Implementacja env, modelu robopsa i algorytmu PPO w kolejnych krokach.

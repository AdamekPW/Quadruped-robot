# RoboDog 🐕‍🦺

Praca magisterska: **nauka chodzenia robopsa metodą uczenia przez wzmacnianie (Reinforcement Learning)**.

## Stack technologiczny

- **Python 3.12** (środowisko zarządzane przez **Condę** — Miniforge)
- **PyTorch 2.11 (CUDA 12.8 / cu128)** — sieci neuronowe i trening
- **MuJoCo** — symulator fizyki
- **Gymnasium** — API środowisk RL
- **Algorytm:** własna implementacja **PPO** w PyTorch
- **GPU:** NVIDIA RTX 5060 Ti (architektura Blackwell, `sm_120`)

> Build `cu128` nie jest przypadkowy: karty Blackwell wymagają CUDA ≥ 12.8, na
> starszych buildach PyTorch w ogóle nie wystartuje.

## Struktura projektu

```
RoboDog/
├── configs/            # Konfiguracje eksperymentów (YAML): hiperparametry, ustawienia env
├── assets/             # Model robopsa dla MuJoCo
│   ├── robodog.xml     #   Definicja MJCF (geometria, stawy, siłowniki)
│   └── meshes/         #   Siatki 3D i tekstury
├── src/robodog/        # Główny pakiet (instalowalny: pip install -e .)
│   ├── envs/           #   Środowiska Gymnasium owijające MuJoCo (obserwacje, akcje, nagroda)
│   ├── algorithms/     #   Własne implementacje RL (PPO) + sieci actor/critic
│   └── utils/          #   Wczytywanie konfiguracji, logowanie, seedy
├── scripts/            # Punkty wejścia CLI: train.py, evaluate.py, record_video.py
├── notebooks/          # Analiza wyników, wykresy krzywych uczenia do pracy
├── tests/              # Testy środowiska (kształty obserwacji, granice akcji)
├── runs/               # Logi / TensorBoard (ignorowane w git)
├── checkpoints/        # Zapisane modele (ignorowane w git)
├── environment.yml     # Definicja środowiska Conda — źródło prawdy o zależnościach
└── pyproject.toml      # Metadane pakietu (instalacja edytowalna)
```

## Instalacja

Wymaga Condy. Jeśli jeszcze jej nie masz, zainstaluj **Miniforge** (dystrybucja
oparta o kanał `conda-forge`, bez ograniczeń licencyjnych kanału `defaults`
Anacondy — istotne przy użyciu na uczelni):

```bash
curl -L -o miniforge.sh \
  "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
bash miniforge.sh -b -p "$HOME/miniforge3"
"$HOME/miniforge3/bin/conda" init bash    # dopisuje aktywację do ~/.bashrc
exec bash                                  # przeładuj powłokę
```

Następnie środowisko projektu:

```bash
conda env create -f environment.yml
conda activate robodog
pip install -e .
```

Sprawdzenie, czy GPU jest widoczne:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Po zmianie `environment.yml`:

```bash
conda env update -f environment.yml --prune
```

### Dlaczego torch instaluje się pipem, skoro używamy Condy?

Projekt PyTorch **nie publikuje już paczek dla Condy** — kanał `pytorch` został
wygaszony, a jedyną oficjalną drogą są koła (wheels) z indeksu PyTorcha. Dlatego
Conda zarządza Pythonem i środowiskiem, a sam torch dociąga pip z sekcji `pip:`
w `environment.yml`. To układ zalecany przez samo PyTorch i standardowo
akceptowany na klastrach obliczeniowych.

## Demo

Dwa skrypty do oswojenia się z narzędziami (używają `Ant-v5` — czworonoga
dostarczanego z Gymnasium, jako tymczasowego zamiennika robopsa):

```bash
python scripts/demo_gym_ant.py        # pętla RL reset() -> step() z losową polityką
python scripts/demo_mujoco_viewer.py  # interaktywny viewer MuJoCo
```

## Testy

```bash
pytest
```

## Status

🚧 Projekt w fazie inicjalizacji — struktura szkieletowa. Implementacja env,
modelu robopsa i algorytmu PPO w kolejnych krokach.

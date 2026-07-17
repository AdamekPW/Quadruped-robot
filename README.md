# RoboDog 🐕‍🦺

Praca magisterska: **nauka chodzenia robopsa metodą uczenia przez wzmacnianie (Reinforcement Learning)**.

## Stack technologiczny

- **Python 3.12** (środowisko zarządzane przez **Condę** — Miniforge)
- **PyTorch 2.11 (CUDA 12.8 / cu128)** — sieci neuronowe i trening
- **mjlab** — framework RL: API w stylu Isaac Lab + **MuJoCo Warp** (symulacja
  tysięcy środowisk równolegle na GPU). Patrz [docs/mjlab-architektura.md](docs/mjlab-architektura.md).
- **MuJoCo** — symulator fizyki
- **Algorytm:** własna implementacja **PPO** w PyTorch (wkład własny pracy;
  `rsl-rl` z mjlab służy jako punkt odniesienia — patrz „Plan" niżej)
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
│   ├── envs/           #   Konfiguracje środowisk mjlab (obserwacje, nagrody, zdarzenia)
│   ├── algorithms/     #   Własne implementacje RL (PPO) + sieci actor/critic
│   └── utils/          #   Wczytywanie konfiguracji, logowanie, seedy
├── scripts/            # Punkty wejścia CLI (docelowo: własny train.py z naszym PPO)
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

# WYMAGANE: obejście regresji w mujoco-warp 3.10.0.2, która wywala trening
# przy num_envs >= ~176. Szczegóły: docs/mjlab-architektura.md, sekcja 6.
pip install --no-deps mujoco-warp==3.10.0.1
```

Poprawność instalacji sprawdza `pytest` (patrz „Testy") — w tym obecność
właściwej wersji `mujoco-warp`.

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

## Trening

```bash
# Trening referencyjny (baseline) na gotowym PPO z rsl-rl:
python -m mjlab.scripts.train Mjlab-Velocity-Flat-Unitree-Go1 \
  --env.scene.num-envs 4096 \
  --agent.logger tensorboard

# Podgląd nauczonej polityki:
python -m mjlab.scripts.play Mjlab-Velocity-Flat-Unitree-Go1 --help
```

> `--agent.logger tensorboard` jest świadome: domyślnym loggerem mjlab jest
> **wandb**, który wysyła przebiegi treningu na zewnętrzny serwis i wymaga konta.

## Testy

```bash
pytest
```

## Plan

Cel: **Unitree Go2** (wymóg promotora) chodzący dzięki **własnej implementacji PPO**.
Droga do tego jest podzielona na etapy — orientacyjne, nie sztywne:

1. ✅ Środowisko (Conda, CUDA na Blackwellu, mjlab) — działa, potwierdzone testami.
2. 🚧 **Baseline Go1 na `rsl-rl`** — mjlab dostarcza Go1 od ręki. Daje punkt
   odniesienia: wiadomo, jak wygląda poprawnie nauczony chód i ile to trwa.
3. ⬜ **Port Go2** — mjlab **nie ma Go2**; model trzeba przenieść z
   [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).
   Szczegóły: [docs/mjlab-architektura.md](docs/mjlab-architektura.md), sekcja 5.
4. ⬜ **Własny PPO** — podmiana `rsl-rl` na własną implementację i porównanie z
   baseline'em. To jest wkład własny pracy.

## Status

🚧 Środowisko gotowe i zweryfikowane. Trening baseline'u w toku.

# mjlab — co za co odpowiada i dlaczego

Notatka orientacyjna: mjlab jest cudzą biblioteką, więc ten dokument nie opisuje
naszego kodu, tylko **mapę tego, w co się wpinamy**. Bez niej konfiguracje Go1
wyglądają jak ściana magicznych stałych.

## 1. Po co nam w ogóle mjlab

Do tej pory projekt zakładał: Gymnasium + zwykłe MuJoCo = **jedno** środowisko
liczone na CPU. Nauka chodzenia metodą RL wymaga rzędu **setek milionów kroków
symulacji**. Na jednym środowisku to tygodnie liczenia.

mjlab skleja dwie rzeczy:

| Składnik | Co robi |
|---|---|
| **MuJoCo Warp** (`mujoco-warp`) | MuJoCo przepisane na GPU. Symuluje tysiące kopii robota **równolegle**, zamiast jednej po drugiej. |
| **API w stylu Isaac Lab** | Sposób opisywania środowiska z klocków (*manager-based API*) zamiast jednej wielkiej klasy. |

Efekt: to, co na CPU trwałoby tygodnie, na Twoim GPU schodzi do godzin.
To jedyny powód, dla którego rezygnujemy z własnych wrapperów Gymnasium.

> **Warp** (`warp-lang`) to biblioteka NVIDII do pisania kerneli GPU w Pythonie.
> MuJoCo Warp to MuJoCo zaimplementowane w Warpie. Stąd wymóg karty NVIDIA —
> mjlab bez GPU trenować nie będzie (macOS obsługiwany tylko do ewaluacji).

## 2. Manager-based API — sedno, które trzeba zrozumieć

Klasyczne środowisko Gymnasium to jedna klasa z metodami `reset()` i `step()`,
w której funkcja nagrody, obserwacje i warunki końca epizodu są wymieszane w kodzie.

mjlab rozbija to na **osobne, konfigurowalne kawałki** (`managers`):

| Manager | Odpowiada za | Przykład dla czworonoga |
|---|---|---|
| **Observation** | co robot „widzi" | kąty stawów, prędkości, żyroskop, zadana prędkość |
| **Reward** | za co dostaje punkty | nagroda za trzymanie zadanej prędkości, kara za zużycie energii |
| **Termination** | kiedy epizod się kończy | robot się przewrócił, upłynął limit czasu |
| **Event** | co losowo zaburzamy | losowe tarcie, masa, popchnięcie robota |
| **Action** | jak wyjście sieci steruje robotem | `JointPositionActionCfg` — sieć zadaje docelowe kąty stawów |
| **Command** | co robot ma zrobić | `UniformVelocityCommandCfg` — losowa zadana prędkość |
| **Curriculum** | jak trudność rośnie w czasie | teren coraz bardziej wyboisty |

**Dlaczego to dobre dla pracy magisterskiej:** funkcja nagrody przestaje być
zaszyta w kodzie i staje się listą nazwanych składników z wagami. Możesz
wyłączyć jeden składnik i pokazać w pracy, jak to zmieniło chód — to gotowy
materiał na rozdział o badaniu ablacyjnym (*ablation study*: wyłączamy element,
patrzymy co się psuje).

> **Ważne — `Action` to nie moment siły.** W `JointPositionActionCfg` sieć nie
> zadaje momentów w stawach, tylko **docelowe kąty**, które regulator PD zamienia
> na momenty. To standard w lokomocji nóg i częsty punkt nieporozumień: wyjście
> sieci ma jednostkę „kąt", nie „siła".

## 3. Mapa katalogów mjlab

```
src/mjlab/
├── asset_zoo/robots/      # Modele robotów: MJCF + stałe (nazwy stawów, poza domyślna)
│   ├── unitree_go1/       #   Czworonóg — TEN mamy
│   ├── unitree_g1/        #   Humanoid
│   └── i2rt_yam/          #   Ramię manipulacyjne
│                          #   UWAGA: Go2 TU NIE MA — patrz sekcja 5
├── tasks/                 # Zadania
│   ├── velocity/          #   Chodzenie z zadaną prędkością — TO nas interesuje
│   │   └── config/go1/    #     env_cfgs.py (środowisko) + rl_cfg.py (PPO)
│   ├── tracking/          #   Naśladowanie ruchu z nagrania
│   ├── manipulation/      #   Chwytanie
│   └── registry.py        #   Rejestr zadań (nadaje im ID typu "Mjlab-...")
├── managers/              # Implementacje managerów z sekcji 2
├── envs/                  # ManagerBasedRlEnv — klasa spinająca managery
└── rl/                    # Most do rsl-rl (PPO)
```

## 4. Co robią dwa pliki konfiguracji Go1

To są pliki, które przy porcie Go2 będziemy pisać od nowa — warto rozumieć podział.

### `env_cfgs.py` — opis **świata i zadania**

Definiuje fizykę, robota, czujniki i nagrody. Fragmenty, które warto znać:

- `cfg.scene.entities = {"robot": get_go1_robot_cfg()}` — wstawia robota do sceny.
- `cfg.sim.mujoco.cone = "elliptic"`, `impratio = 10` — ustawienia modelu tarcia.
  Domyślny stożek tarcia (*friction cone*) MuJoCo jest piramidalny i szybszy, ale
  eliptyczny jest dokładniejszy. Przy stopach robota to ma znaczenie: zbyt
  „ślizgający się" kontakt uczy robota chodu, który nie przeniesie się na
  prawdziwy sprzęt.
- `ContactSensorCfg(...)` — czujnik kontaktu stopa-podłoże. `foot_names = ("FR",
  "FL", "RR", "RL")` to nogi: przód-prawa, przód-lewa, tył-prawa, tył-lewa.
- `TerrainHeightSensorCfg` / `RayCastSensorCfg` — „skan" wysokości terenu wokół
  robota (promienie w dół). Dzięki temu polityka wie, że przed nią jest schodek.

### `rl_cfg.py` — opis **algorytmu uczenia**

To jest ten plik, który przy podmianie na Twój PPO stanie się nieaktualny.
Obecnie zwraca konfigurację dla rsl-rl:

- `actor` / `critic`: `hidden_dims=(512, 256, 128)`, aktywacja `elu`.
  Dwie sieci — **actor** wybiera akcje, **critic** ocenia, jak dobra jest sytuacja.
- `clip_param=0.2` — słynne „przycięcie" PPO: ogranicza, jak bardzo polityka może
  się zmienić w jednym kroku. Serce algorytmu.
- `gamma=0.99` — współczynnik dyskontowania (jak bardzo liczy się przyszłość).
- `lam=0.95` — parametr GAE (sposób szacowania przewagi akcji).
- `entropy_coef=0.01` — premia za losowość, wymusza eksplorację.
- `desired_kl=0.01` + `schedule="adaptive"` — learning rate **sam się dostraja**,
  żeby zmiana polityki na krok trzymała zadaną „odległość" KL.
- `num_steps_per_env=24`, `max_iterations=10_000` — długość zbierania danych i
  liczba iteracji treningu.

**To jest Twoja lista odniesienia.** Gdy będziesz podpinać własny PPO, te wartości
są punktem startowym — jeśli Twoja implementacja z tymi samymi
hiperparametrami uczy się wyraźnie gorzej, to sygnał, że masz błąd, a nie że
„PPO nie działa". Dokładnie po to robimy najpierw baseline na rsl-rl.

## 5. Go2 — czego brakuje i co trzeba dorobić

Promotor wskazał **Go2**, a mjlab dostarcza tylko **Go1**. Model Go2 jest w
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(katalog `unitree_go2`). Port będzie wymagał:

1. `asset_zoo/robots/unitree_go2/xmls/go2.xml` + siatki — przeniesienie z Menagerie.
2. `go2_constants.py` — nazwy stawów, poza domyślna, parametry siłowników,
   `GO2_ACTION_SCALE`.
3. `tasks/velocity/config/go2/env_cfgs.py` + `rl_cfg.py` — po wzorze Go1.
4. Rejestracja zadania `Mjlab-Velocity-Flat-Unitree-Go2` w `__init__.py`.

Go1 i Go2 to bardzo podobne czworonogi (12 stawów, 4 nogi), różnią się wymiarami,
masami i siłownikami — dlatego Go1 jest sensownym etapem pośrednim, a nie
straconym czasem. Nazwy nóg (`FR/FL/RR/RL`) i struktura zadania przenoszą się 1:1.

## 6. Pułapka: regresja w mujoco-warp 3.10.0.2

**To jest najważniejsza rzecz w tym dokumencie.** Jeśli trening nagle przestanie
działać po odbudowie środowiska, zacznij tutaj.

### Objaw

Trening zadania `velocity` wywala się z `CUDA error 700: an illegal memory
access`, zanim policzy choćby jedną iterację. W logu widać setki linii
`Warp CUDA error 700 (in function wp_free_device_async)` — to **kaskada przy
sprzątaniu pamięci**, nie przyczyna. Prawdziwy błąd jest na samej górze logu.

### Diagnoza (zmierzona na RTX 5060 Ti, sm_120)

| Konfiguracja | Wynik |
|---|---|
| Prosty kernel Warp | działa — sam Warp na Blackwellu jest sprawny |
| `Mjlab-Cartpole-Balance` @ 4096 env | **działa** |
| `Mjlab-Velocity-Flat-Unitree-Go1` @ ≤ 160 env | działa |
| `Mjlab-Velocity-Flat-Unitree-Go1` @ ≥ 176 env | **pada** |
| `Mjlab-Velocity-Flat-Unitree-G1` @ 512 env | **pada** (więc to nie wina Go1) |
| Go1 @ 4096 na `mujoco-warp==3.10.0.1` | **działa** |

Ścieżka awarii (widoczna dopiero z `CUDA_LAUNCH_BLOCKING=1`, bo błędy CUDA są
asynchroniczne i domyślny ślad stosu wskazuje złe miejsce):

```
event_manager.apply("startup")     # zdarzenia randomizacji dziedziny
  -> sim.recompute_constants()     # bo randomizacja zmieniła rozkład mas
    -> mujoco_warp io.set_const()
      -> smooth.crb()              # macierz mas (composite rigid body)
        -> kernel _M               # <- tu wybucha
```

Cartpole przechodzi, bo **nie ma zdarzeń randomizacji fizyki** — nie wchodzi na
tę ścieżkę. To nie jest brak pamięci (przy awarii wolne było ~15,8 GiB) ani zbyt
małe bufory (`njmax`, `contact_sensor_maxmatch` — sprawdzone, podniesienie nie
pomaga).

### Obejście

```bash
pip install --no-deps mujoco-warp==3.10.0.1
```

`--no-deps` jest konieczne, bo mjlab pinuje `mujoco-warp>=3.10.0.2` i pip
odmawia rozwiązania (`ResolutionImpossible`). Z tego samego powodu **nie da się
tego zapisać w `environment.yml`** — jest tam tylko komentarz, a pilnuje tego
test `tests/environment_test.py::test_mujoco_warp_has_no_known_regression`.

Po obejściu `pip check` zgłasza złamany pin mjlab — to oczekiwane i nieszkodliwe.

### Status

Na dzień 2026-07-17 **to nie jest zgłoszony błąd** — przeszukane issues w
`google-deepmind/mujoco_warp` i `mujocolab/mjlab` nie zawierają tego przypadku.
Podobne, ale **inne**: mujoco_warp #1280 (Blackwell, ale modele mięśniowo-ścięgnowe),
mjlab #576 (Blackwell, ale OOM przy 50 000 env). Kandydat do zgłoszenia upstream.

## Źródła

- Praca: [mjlab: A Lightweight Framework for GPU-Accelerated Robot Learning](https://arxiv.org/abs/2601.22074) (arXiv 2601.22074)
- Repo: <https://github.com/mujocolab/mjlab>
- Dokumentacja: <https://mujocolab.github.io/mjlab/>

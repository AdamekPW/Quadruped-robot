# Idiomy — katalog „przed → po"

Konkretne wzorce do wykorzystania w uwagach, na przykładach z tego projektu (mjlab / rsl-rl / PyTorch). Znacznik przy nagłówku = domyślny priorytet z SKILL.md.

## 🔴 Gołe `except` i połykanie błędów
```python
# przed
try:
    runner.load(path)
except:
    print("nie udało się")
# po
try:
    runner.load(path)
except (FileNotFoundError, KeyError) as err:
    raise RuntimeError(f"Nie wczytano checkpointu {path!r}") from err
```
`from err` zachowuje oryginalny stack trace (odpowiednik `throw;` zamiast `throw ex;` w C#).

## 🔴 Magiczne liczby w pętli treningowej
```python
# przed
if it % 50 == 0 and it > 200:
# po (stała modułowa, U GÓRY pliku)
LOG_INTERVAL = 50
WARMUP_ITERATIONS = 200
```
W treningu to nie kosmetyka: te liczby trafiają do opisu eksperymentu w pracy i muszą być w jednym miejscu.

## 🔴 Funkcja robiąca cztery rzeczy
Sygnał: `__init__` na 60 linii, w którym jest wczytywanie configu + budowa sieci + ładowanie checkpointu + logowanie. Rozbij na prywatne metody `_build_networks()`, `_load_teacher()`. Kryterium: czy da się nazwać fragment jednym czasownikiem.

## 🟡 Guard clause zamiast zagnieżdżeń
```python
# przed
if cfg is not None:
    if cfg.enabled:
        for k in keys:
            ...
# po
if cfg is None or not cfg.enabled:
    return
for k in keys:
    ...
```
Ta sama zasada co w C#, ale w Pythonie zagnieżdżenia bolą bardziej — wcięcia są składnią, więc czwarty poziom naprawdę przestaje się czytać.

## 🟡 Comprehension zamiast `append` w pętli
```python
# przed
dims = []
for layer in layers:
    if layer.trainable:
        dims.append(layer.out_dim)
# po
dims = [layer.out_dim for layer in layers if layer.trainable]
```
To LINQ `Where().Select()`. Granica: jeśli comprehension ma dwa `for` i `if`, albo nie mieści się w linii — wróć do pętli. Zagnieżdżone comprehension to nie punkt honoru.

## 🟡 `dataclass` zamiast worka argumentów
```python
# przed
def build(hidden, activation, lr, epochs, obs_norm, device): ...
# po
@dataclass(frozen=True)
class TrainingCfg:
    hidden_dims: tuple[int, ...] = (512, 256, 256)
    activation: str = "elu"
    learning_rate: float = 1e-3
def build(cfg: TrainingCfg): ...
```
Odpowiednik `record` z C#. `frozen=True` chroni przed przypadkową mutacją współdzielonego konfigu (patrz pułapka „wszystko jest referencją”). Wariant robisz przez `dataclasses.replace(cfg, learning_rate=3e-4)`.

## 🟡 `logging` zamiast `print`
```python
# przed
print(f"[INFO] Załadowano teachera z: {path}")
# po
logger = logging.getLogger(__name__)
logger.info("Załadowano teachera z: %s", path)
```
Zysk: da się wyciszyć/przekierować bez ruszania kodu, a przy tysiącach środowisk `print` potrafi zalać terminal. Wyjątek: skrypty w `scripts/` — tam `print` jest w porządku, to interfejs użytkownika.

## 🟡 `pathlib` zamiast sklejania stringów
```python
# przed
ckpt = log_dir + "/" + run + "/model.pt"
# po
ckpt = Path(log_dir) / run / "model.pt"
```
`Path` to `FileInfo`/`Path.Combine` w jednym. Ma `.exists()`, `.stem`, `.parent`, `.glob("*.pt")` — i działa niezależnie od systemu.

## 🟡 `with` zamiast ręcznego sprzątania
```python
with torch.no_grad():
    actions = policy(obs)
```
To `using`. Jeśli piszesz parę „ustaw stan → przywróć stan" (tryb ewaluacji, seed, plik) — zrób `@contextlib.contextmanager`, nie `try/finally` w każdym wywołaniu.

## 🟡 Rozpakowywanie zamiast indeksów
```python
# przed
batch_size = shape[0]; height = shape[2]; width = shape[3]
# po
batch, channels, height, width = depth.shape
```
Nazwane wymiary tensora są w tym repo najtańszą obroną przed błędem kształtu.

## 🟡 `enumerate` / `zip` zamiast pętli po indeksach
```python
for i, layer in enumerate(layers): ...
for name, tensor in zip(names, tensors, strict=True): ...
```
`strict=True` (3.10+) rzuca wyjątek przy różnych długościach — bez tego `zip` po cichu ucina do krótszej. Warto zawsze.

## 🟡 Typuj publiczne API modułu
```python
def depth_image(env: ManagerBasedRlEnv, sensor_name: str, cutoff: float) -> torch.Tensor:
```
Reguła kciuka na ten projekt: **funkcje wywoływane spoza pliku — typuj; lokalne helpery i skrypty — nie musisz.** Argumenty przyjmuj szeroko (`Sequence[float]`), zwracaj konkretnie (`list[float]`).

## 🟡 `assert` do niezmienników, wyjątek do danych z zewnątrz
```python
assert depth is not None, f"Kamera {name!r} nie zwraca głębi."
```
`assert` mówi „to nie powinno się zdarzyć nigdy" i informuje pyrighta, że dalej nie ma `None`. Uwaga: `python -O` je wyłącza, więc **nie** waliduj nimi argumentów od użytkownika — tam `raise ValueError`.

## 🟡 Generator zamiast budowania wielkiej listy
```python
# przed
frames = [render(i) for i in range(10_000)]
# po
def frames(n: int) -> Iterator[Frame]:
    for i in range(n):
        yield render(i)
```
Dokładnie `yield return` / `IEnumerable<T>`. Sens ma przy dużych sekwencjach — przy dwudziestu elementach lista jest czytelniejsza.

## ⚪ Świadomie pomijamy w tym projekcie
- Docstringi do prywatnych helperów (`_foo`) — nazwa wystarczy.
- `Protocol`/ABC przy jednej implementacji — dodaj dopiero przy drugiej.
- `__slots__`, `functools.cache`, mikrooptymalizacje — wąskim gardłem jest GPU, nie Python.
- Testy jednostkowe konfiguracji. Testujemy to, co liczy (transformacje obserwacji, kształty), nie to, co deklaruje.

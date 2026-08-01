# Mapa C# → Python

Ściąga do budowania mostów. Część I to odpowiedniki (co znasz z C#, jak się nazywa tutaj), część II — miejsca, gdzie intuicja z C# **aktywnie szkodzi**. Część II jest ważniejsza.

## I. Odpowiedniki

| C# | Python | Uwaga |
|---|---|---|
| `namespace` | moduł = plik `.py`, pakiet = katalog | Import to `import robodog.tasks`, nie `using` |
| `static class Helpers` | zwykły moduł z funkcjami | Klasa z samymi `@staticmethod` to zapach kodu |
| `record Point(int X, int Y)` | `@dataclass(frozen=True)` | `frozen=True` ≈ `init`-only properties |
| `class` z propertiesami | `@dataclass` | Bez `dataclass` musisz ręcznie pisać `__init__`, `__eq__`, `__repr__` |
| `{ get; }` / `=> expr` | `@property` | Tylko gdy jest logika; zwykły publiczny atrybut jest OK i **nie** wymaga property |
| `interface IFoo` | `typing.Protocol` (strukturalnie) albo `abc.ABC` (nominalnie) | `Protocol` = duck typing sprawdzany przez pyright, bez dziedziczenia |
| `abstract class` | `abc.ABC` + `@abstractmethod` | |
| `IEnumerable<T>` | `Iterable[T]` | Do argumentów bierz `Iterable`/`Sequence`, zwracaj konkretną `list` |
| `yield return` | `yield` (generator) | Prawie identyczna semantyka leniwa |
| LINQ `Where/Select` | list/dict/set comprehension | `[f(x) for x in xs if p(x)]` |
| LINQ `Aggregate/Any/All/Zip` | `functools.reduce`, `any`, `all`, `zip` | `itertools` = reszta LINQ (`chain`, `groupby`, `islice`) |
| `IDisposable` + `using` | context manager + `with` | `__enter__`/`__exit__` lub `@contextlib.contextmanager` |
| `try/catch/finally` | `try/except/finally` | Jest też `else` (gdy nie było wyjątku) |
| `throw new ArgumentException` | `raise ValueError(...)` | `ValueError` = zła wartość, `TypeError` = zły typ |
| atrybuty `[Obsolete]` | dekoratory `@deprecated` | Dekorator to funkcja opakowująca funkcję — działa w runtime, nie w metadanych |
| `T?` / `Nullable<T>` | `X | None` | Nie ma operatorów `?.` ani `??=`; `??` ≈ `or` (uwaga na falsy!) |
| `var` | wnioskowanie typu (brak słowa kluczowego) | Adnotacje są opcjonalne i **nie** działają w runtime |
| generics `List<T>` | `list[T]` | Tylko dla type checkera — nie ma reifikacji, `isinstance(x, list[int])` nie zadziała |
| `$"tekst {x}"` | f-string `f"tekst {x}"` | `f"{x=}"` drukuje `x=5`, `f"{x!r}"` = `ToString()` debugowy |
| `ToString()` | `__str__` (dla ludzi) / `__repr__` (dla debugu) | Jeśli piszesz tylko jeden — pisz `__repr__` |
| `Equals` + `GetHashCode` | `__eq__` + `__hash__` | `@dataclass` generuje za darmo; `frozen=True` daje też hash |
| `IComparable` / `OrderBy` | `sorted(xs, key=...)` | `key` zamiast comparatora — prościej niż w C# |
| `enum` | `enum.Enum` / `StrEnum` | |
| `internal` / `private` | prefiks `_nazwa` (konwencja) | Nie ma egzekwowania — to umowa, nie kompilator |
| `readonly` | `Final[...]` (pyright) / `frozen=True` | W runtime nic nie chroni |
| NuGet | pip / conda | Tu: zależności w `environment.yml` |
| xUnit `[Fact]` | pytest: funkcja `def test_x()` | Bez klas, bez atrybutów, `assert` zwykły |
| DI container | przekazywanie funkcji/obiektów w argumentach | Kontenery DI są w Pythonie rzadkością i zwykle przerostem formy |

## II. Gdzie intuicja z C# prowadzi na minę

### 1. Mutowalne argumenty domyślne 🔴
```python
def f(items: list[int] = []):   # ŹLE — ta lista powstaje RAZ, przy definicji
    items.append(1)
```
Domyślna wartość jest tworzona przy definiowaniu funkcji, nie przy wywołaniu — kolejne wywołania współdzielą jeden obiekt. Poprawnie: `items: list[int] | None = None` i `items = items or []` w środku. **W C# to nawet się nie skompiluje** (domyślne muszą być stałymi), więc nie masz na to odruchu.

### 2. Wszystko jest referencją, nie ma `struct` 🔴
```python
cfg_b = cfg_a          # ten sam obiekt, nie kopia
cfg_b = copy.copy(a)   # płytka kopia
cfg_b = copy.deepcopy(a)  # głęboka
```
W C# przypisanie `struct`/`record struct` kopiuje. Tu **nigdy**. Częsty błąd w tym repo: modyfikacja współdzielonego konfigu środowiska psuje inny wariant. Stąd `dataclasses.replace(cfg, terrain=...)` zamiast `cfg.terrain = ...`.

### 3. `is` to nie `==` 🔴
`is` porównuje tożsamość (referencję) — to `ReferenceEquals`. `==` woła `__eq__`. Używaj `is` **tylko** z `None`, `True`, `False`. `if x == None` działa, ale zdradza brak nawyku.

### 4. Nie ma przeciążania metod
Jedna nazwa = jedna funkcja; druga definicja **cicho nadpisuje** pierwszą. Zamiast overloadów: argumenty domyślne, `*args`, albo `functools.singledispatch`. Konstruktory alternatywne (`Foo.FromFile(...)`) robi się jako `@classmethod def from_file(cls, ...)`.

### 5. Brak kompilatora — pyright to jedyna siatka
Adnotacje typów są **ignorowane w runtime**. Literówka w nazwie atrybutu wybuchnie dopiero po 40 minutach treningu. Praktyczny wniosek: typuj publiczne funkcje i uruchamiaj pyright — to twój `csc`.

### 6. Truthiness zamiast `!= null`
`if x:` jest fałszywe dla `None`, `0`, `0.0`, `""`, `[]`, `{}`. Czyli `if not tensor_list:` łapie też pustą listę, a `x or default` podmieni poprawne `0`. Gdy chodzi o „czy podano" — pisz `if x is None`.

### 7. Wyjątki są tanie — EAFP zamiast LBYL
W C# rzucanie wyjątków jest kosztowne, więc sprawdzasz warunki wcześniej. W Pythonie idiomatyczne jest „spróbuj i złap":
```python
try:
    return cache[key]
except KeyError:
    ...
```
Ale: `except Exception:` bez potrzeby (a już zwłaszcza gołe `except:`) to 🔴 — łyka też `KeyboardInterrupt` i literówki.

### 8. Domknięcia łapią zmienną, nie wartość
```python
fns = [lambda: i for i in range(3)]   # wszystkie zwrócą 2
fns = [lambda i=i: i for i in range(3)]  # obejście
```
C# ma ten sam problem z `for` (przed C# 5) — ale ty pewnie pamiętasz go już naprawionego.

### 9. `/` to zawsze float
`7 / 2 == 3.5`, nawet dla `int`. Dzielenie całkowite to `//`. W kodzie liczącym indeksy/rozmiary siatek (np. wymiar po `Conv2d`) to źródło cichych błędów typu „float jako indeks".

### 10. Wątki nie dają równoległości (GIL)
`threading` nadaje się tylko do I/O. Do CPU — `multiprocessing` albo biblioteka natywna (torch i tak zwalnia GIL wewnątrz). `async/await` istnieje i wygląda jak w C#, ale to jednowątkowa pętla zdarzeń — nie `Task.Run`.

### 11. Nie ma metod rozszerzających ani `partial`
Nie da się dokleić metody do cudzej klasy (poza brzydkim monkey-patchingiem). Zamiast tego: zwykła funkcja w module przyjmująca obiekt jako pierwszy argument. To dlatego Python ma `len(x)`, a nie `x.Length`.

### 12. Dziedziczenie jest wielokrotne i ma MRO
`super().__init__(...)` idzie po liniowym porządku klas (MRO), nie po „klasie bazowej". W tym repo ma to znaczenie przy dziedziczeniu po klasach rsl-rl — zmiana kolejności baz zmienia zachowanie.

### 13. Właściwości nie muszą istnieć od początku
Obiekt to worek atrybutów; można dopisać nowy w dowolnym momencie (`self.foo = 1` w losowej metodzie). Da się, ale **nie należy** — deklaruj wszystko w `__init__`, inaczej pyright i czytelnik nie mają szans.

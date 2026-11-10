# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "wandb==0.29.0",
#     "wandb-workspaces==0.4.11",
# ]
# ///
"""Generator raportu wandb z wykresami zbiorczymi dla treningu RoboDoga.

Po co: wandb domyślnie robi JEDEN panel na metrykę, więc `Curriculum/terrain_levels/*`
rozsypuje się na dziewięć osobnych wykresów. Ten skrypt składa je w jeden panel,
na którym widać naraz, jak robot radzi sobie na każdym typie terenu.

Definiowanie wykresów kodem (zamiast klikania w UI) daje dwie rzeczy: raport
odtwarza się identycznie przy każdym eksperymencie, a do pracy magisterskiej
można napisać, JAK powstał wykres, zamiast opisywać kliknięcia.

UWAGA — INNE ŚRODOWISKO NIŻ TRENING.
`wandb-workspaces` wymaga wandb >= 0.28.2, a środowisko treningowe ma
przypięte 0.28.1. Żeby go nie ruszać, skrypt deklaruje własne zależności
w bloku `# /// script` (PEP 723) na górze pliku. `uv run` widzi ten blok,
pomija zależności projektu i buduje dla skryptu osobne, zapamiętane w cache
środowisko.

Dwa cele, bo służą do czego innego:

  * `--target workspace` (domyślne) — zapisany widok w pulpicie projektu.
    TO JEST TO, NA CO PATRZYSZ W TRAKCIE TRENINGU: wykresy aktualizują się
    na żywo, widok wybiera się z rozwijanej listy u góry projektu.
  * `--target report` — dokument z opisami, do pracy magisterskiej.
    Nie aktualizuje się sam na oczach, trzeba odświeżyć stronę.

Użycie:

    uv run scripts/wandb_report.py
    uv run scripts/wandb_report.py --target report
    uv run scripts/wandb_report.py --target both --dry-run
"""

from __future__ import annotations

import argparse

import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

ENTITY = "szulczynskiadam31-poznan-university-of-technology"
PROJECT = "robodog"
# Nazwa zapisanego widoku workspace'u (rozwijana lista u góry projektu).
WORKSPACE_NAME = "RoboDog — zbiorczo"

# Typy terenu w ROUGH_TERRAINS_CFG, w kolejności kolumn siatki. Muszą zgadzać się
# z kluczami, które `terrain_levels_vel` wypuszcza do loga — mjlab nazywa je
# wprost po nazwach sub-terenów z konfiguracji generatora.
TERRAIN_TYPES = (
    "flat",
    "pyramid_stairs",
    "pyramid_stairs_inv",
    "hf_pyramid_slope",
    "hf_pyramid_slope_inv",
    "random_rough",
    "wave_terrain",
)

# Liczba wierszy siatki terenu = liczba poziomów trudności (patrz `as_play`).
# Poziom 0 to najłatwiejszy wiersz, `NUM_TERRAIN_ROWS - 1` najtrudniejszy.
NUM_TERRAIN_ROWS = 10

_CURRICULUM_PREFIX = "Curriculum/terrain_levels"
_MASTERY_PREFIX = "Curriculum/terrain_mastery"


def terrain_levels_panel() -> wr.LinePlot:
    """Jeden wykres z poziomem curriculum dla każdego typu terenu osobno.

    `max` celowo pomijamy: to maksimum po wszystkich środowiskach, więc gdy
    JEDEN robot dobije do ostatniego wiersza, krzywa przykleja się do sufitu
    i tylko zjada zakres osi. `mean` zostaje jako linia odniesienia.
    """
    # GRUPA NIEPRZECHWYTUJĄCA `(?:...)` JEST OBOWIĄZKOWA. Wandb tworzy OSOBNY
    # panel dla każdej wartości złapanej przez grupę przechwytującą — czyli
    # zwykłe `(a|b)` rozbiłoby to z powrotem na osiem wykresów, dokładnie to,
    # czego chcemy uniknąć. `(?:a|b)` trzyma wszystko na jednym panelu.
    #
    # Regex jest w składni RE2, która NIE MA lookaheadu — dlatego `max`
    # wykluczamy wyliczając pozostałe klucze, a nie przez `(?!max)`.
    warianty = "|".join(("mean",) + TERRAIN_TYPES)
    return wr.LinePlot(
        title="Poziom curriculum wg typu terenu",
        x="Step",
        metric_regex=f"^{_CURRICULUM_PREFIX}/(?:{warianty})$",
        title_y="średni poziom (0 = najłatwiejszy)",
        range_y=(0.0, float(NUM_TERRAIN_ROWS - 1)),
        legend_position="east",
        smoothing_factor=0.0,
        layout=wr.Layout(w=16, h=10),
    )


def terrain_mastery_panel() -> wr.LinePlot:
    """Odsetek zaliczonych epizodów wg typu terenu — metryka bez sufitu.

    `terrain_levels` nasyca się na 6.0 z 9, bo mjlab przerzuca roboty ze szczytu
    drabiny na losowy poziom. Ta krzywa idzie do 100% i dopiero ona odróżnia
    politykę, która wygrywa wszędzie, od takiej, która przegrywa co piąty epizod.
    """
    warianty = "|".join(("mean",) + TERRAIN_TYPES)
    return wr.LinePlot(
        title="Opanowanie terenu wg typu (% zaliczonych epizodów)",
        x="Step",
        metric_regex=f"^{_MASTERY_PREFIX}/(?:{warianty})$",
        title_y="zaliczone epizody [%]",
        range_y=(0.0, 100.0),
        legend_position="east",
        smoothing_factor=0.0,
        layout=wr.Layout(w=16, h=10),
    )


def terrain_mastery_by_level_panel() -> wr.LinePlot:
    """To samo, ale w rozbiciu na wiersze siatki — pokazuje, GDZIE stoi ściana.

    Krzywe niskich poziomów wysycają się pierwsze; poziom, na którym krzywa
    przestaje rosnąć, to aktualna granica możliwości polityki.
    """
    poziomy = "|".join(f"level_{i}" for i in range(NUM_TERRAIN_ROWS))
    return wr.LinePlot(
        title="Opanowanie terenu wg poziomu trudności",
        x="Step",
        metric_regex=f"^{_MASTERY_PREFIX}/(?:{poziomy})$",
        title_y="zaliczone epizody [%]",
        range_y=(0.0, 100.0),
        legend_position="east",
        smoothing_factor=0.0,
        layout=wr.Layout(w=16, h=10),
    )


def tracking_panel() -> wr.LinePlot:
    """Jak dobrze polityka trafia w zadaną prędkość — błąd liniowy i kątowy."""
    return wr.LinePlot(
        title="Błąd śledzenia komendy prędkości",
        x="Step",
        y=["Episode_Metrics/vel_error_xy", "Episode_Metrics/vel_error_yaw"],
        title_y="błąd [m/s] / [rad/s]",
        legend_position="east",
        layout=wr.Layout(w=8, h=8),
    )


def terminations_panel() -> wr.LinePlot:
    """Z jakiego powodu kończą się epizody — czyli JAK robot przegrywa."""
    return wr.LinePlot(
        title="Powody zakończenia epizodu",
        x="Step",
        y=[
            "Episode_Termination/fell_over",
            "Episode_Termination/trunk_contact",
            "Episode_Termination/time_out",
            "Episode_Termination/nan",
        ],
        title_y="udział epizodów",
        legend_position="east",
        layout=wr.Layout(w=8, h=8),
    )


def build_workspace(view_url: str | None = None) -> ws.Workspace:
    """Zapisany widok workspace'u — to, na co patrzysz W TRAKCIE treningu.

    Raport jest dokumentem, który się otwiera osobno. Workspace to żywy pulpit
    projektu, więc panel zbiorczy musi trafić także tutaj, żeby był pod ręką
    podczas przebiegu. Widok pojawia się w rozwijanej liście u góry projektu.

    `auto_generate_panels=True` zostawia domyślne panele mjlab poniżej — nasza
    sekcja tylko ląduje na wierzchu, nic nie znika.

    Args:
        view_url: adres ISTNIEJĄCEGO widoku do nadpisania. Bez niego powstaje
            NOWY widok — `save()` na świeżo zbudowanym obiekcie nigdy nie
            nadpisuje po nazwie, bo dopasowuje po wewnętrznym ID. Puszczenie
            skryptu trzy razy bez tego argumentu daje trzy widoki o tej samej
            nazwie w rozwijanej liście.
    """
    sekcja = ws.Section(
        name="Postęp treningu (zbiorczo)",
        panels=[
            terrain_mastery_panel(),
            terrain_mastery_by_level_panel(),
            terrain_levels_panel(),
            tracking_panel(),
            terminations_panel(),
        ],
        is_open=True,
    )
    if view_url:
        workspace = ws.Workspace.from_url(view_url)
        workspace.sections = [sekcja]
        return workspace

    return ws.Workspace(
        entity=ENTITY,
        project=PROJECT,
        name=WORKSPACE_NAME,
        auto_generate_panels=True,
        sections=[sekcja],
    )


def build_report(run_filter: str | None) -> wr.Report:
    """Składa raport z panelami zbiorczymi."""
    runset = wr.Runset(entity=ENTITY, project=PROJECT, name="Przebiegi")
    if run_filter:
        runset.query = run_filter

    return wr.Report(
        entity=ENTITY,
        project=PROJECT,
        title="RoboDog — postęp treningu",
        description=(
            "Panele zbiorcze generowane skryptem scripts/wandb_report.py. "
            "Kluczowy wykres: poziom curriculum osobno dla każdego z 7 typów terenu — "
            "rozjazd między krzywymi pokazuje, który teren jest wąskim gardłem."
        ),
        blocks=[
            wr.H1("Curriculum terenu"),
            wr.P(
                "Wysokość krzywej = na jak trudnym wierszu siatki utrzymują się roboty "
                "danego typu terenu. Rozjazd między krzywymi wskazuje wąskie gardło: "
                "jeśli `flat` dobija do maksimum, a `pyramid_stairs_inv` stoi nisko, "
                "robot umie chodzić, ale nie umie schodzić po schodach."
            ),
            wr.PanelGrid(runsets=[runset], panels=[terrain_levels_panel()]),
            wr.H1("Opanowanie terenu"),
            wr.P(
                "Odsetek epizodów, w których robot przeszedł wymagane 4 m, liczony "
                "osobno dla każdej komórki siatki (poziom x typ terenu). W odróżnieniu "
                "od poziomu curriculum ta metryka nie ma sufitu: 100% znaczy, że robot "
                "przechodzi teren na każdym poziomie trudności, łącznie z najwyższym."
            ),
            wr.PanelGrid(
                runsets=[runset],
                panels=[terrain_mastery_panel(), terrain_mastery_by_level_panel()],
            ),
            wr.H1("Jakość sterowania"),
            wr.PanelGrid(
                runsets=[runset],
                panels=[tracking_panel(), terminations_panel()],
            ),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-filter",
        default=None,
        help="Fragment nazwy przebiegu, np. 'rough-cnn'. Bez tego raport bierze wszystkie.",
    )
    parser.add_argument(
        "--target",
        choices=("workspace", "report", "both"),
        default="workspace",
        help=(
            "workspace = zapisany widok do oglądania W TRAKCIE treningu (domyślne), "
            "report = dokument do pracy, both = oba."
        ),
    )
    parser.add_argument(
        "--view-url",
        default=None,
        help=(
            "Adres istniejącego widoku (…?nw=xxxx) do NADPISANIA. Bez tego "
            "powstaje nowy widok obok starego."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Zbuduj, ale NIE zapisuj niczego w wandb.",
    )
    args = parser.parse_args()

    if args.target in ("workspace", "both"):
        workspace = build_workspace(args.view_url)
        if args.dry_run:
            print(f"[dry-run] Zbudowano widok '{workspace.name}' — nie zapisano.")
        else:
            saved = workspace.save()
            print(f"Zapisano widok workspace: {saved.url}")

    if args.target in ("report", "both"):
        report = build_report(args.run_filter)
        if args.dry_run:
            print(f"[dry-run] Zbudowano raport '{report.title}' — nie zapisano.")
        else:
            report.save()
            print(f"Zapisano raport: {report.url}")


if __name__ == "__main__":
    main()

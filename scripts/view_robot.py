"""Podgląd modelu robota w przeglądarce przez viser (ten sam mechanizm co
`mjlab ... play --viewer viser`, ale na gołym pliku MJCF — bez zadania mjlab
ani checkpointu).

Uruchamia serwer viser i wypisuje adres URL. Otwierasz go w przeglądarce
(lokalnie albo przez tunel SSH) i oglądasz model w 3D — obrót kamery, zoom,
panel z geometrią/stawami. Działa na serwerze bez środowiska graficznego,
bo renderowaniem zajmuje się przeglądarka, nie maszyna.

Domyślnie ładuje Silver Badgera; parametrem --model można wskazać inny XML.

Przykłady:
    # Silver Badger, statyczna poza nominalna (keyframe "home")
    python scripts/view_robot.py

    # z symulacją fizyki (robot reaguje na grawitację)
    python scripts/view_robot.py --simulate

    # inny model, inny port
    python scripts/view_robot.py --model .../go1.xml --port 8081

    # sama diagnostyka modelu, bez serwera
    python scripts/view_robot.py --info-only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco

REPO_ROOT = Path(__file__).resolve().parents[1]
# Do samego oglądania używamy pełnego modelu źródłowego (ma scenę, podłogę i
# keyframe "home"). Czysty silver_badger.xml (bez sceny/keyframe) jest pod mjlab.
DEFAULT_MODEL = (
    REPO_ROOT
    / "src/robodog/assets/robots/silver_badger/xmls/silver_badger_source.xml"
)


def parse_args() -> argparse.Namespace:
    """Parsuje argumenty wiersza poleceń."""
    # __doc__ bywa None (np. przy uruchomieniu Pythona z -OO), stąd zabezpieczenie.
    summary = (__doc__ or "Podgląd modelu robota przez viser.").splitlines()[0]
    parser = argparse.ArgumentParser(description=summary)
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Ścieżka do pliku MJCF (domyślnie: Silver Badger).",
    )
    parser.add_argument(
        "--keyframe",
        default="home",
        help="Keyframe na start, albo 'none' by pominąć (domyślnie: 'home').",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Krokuj fizykę (robot reaguje na grawitację). Domyślnie: statycznie.",
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port serwera viser (domyślnie 8080)."
    )
    parser.add_argument(
        "--info-only",
        action="store_true",
        help="Wypisz informacje o modelu i zakończ, bez uruchamiania serwera.",
    )
    return parser.parse_args()


def print_model_info(model: mujoco.MjModel) -> None:
    """Wypisuje skrótowe informacje o modelu (stawy, siłowniki, wymiary)."""
    joint_types = {
        mujoco.mjtJoint.mjJNT_FREE: "free",
        mujoco.mjtJoint.mjJNT_BALL: "ball",
        mujoco.mjtJoint.mjJNT_SLIDE: "slide",
        mujoco.mjtJoint.mjJNT_HINGE: "hinge",
    }
    print(f"Wymiary: nq={model.nq}  nv={model.nv}  nu={model.nu} (siłowniki)")
    print("Stawy:")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        jtype = joint_types[model.jnt_type[i]]
        print(f"  {name or '<base>':16s} {jtype}")


def apply_keyframe(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> None:
    """Ustawia keyframe o danej nazwie; przy 'none'/braku zostaje poza zerowa."""
    if name.lower() == "none":
        return
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
    if key_id < 0:
        print(f"[uwaga] Brak keyframe'u '{name}' — poza zerowa.")
        return
    mujoco.mj_resetDataKeyframe(model, data, key_id)


def main() -> None:
    """Ładuje model i uruchamia interaktywny podgląd viser w przeglądarce."""
    import viser
    from mjviser import Viewer

    args = parse_args()
    if not args.model.exists():
        raise SystemExit(f"Nie znaleziono pliku modelu: {args.model}")

    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    print(f"Załadowano model: {args.model}")
    print_model_info(model)
    if args.info_only:
        return

    apply_keyframe(model, data, args.keyframe)
    mujoco.mj_forward(model, data)

    # step_fn = None -> viewer nie krokuje fizyki (statyczna poza do oglądania).
    # Przy --simulate krokujemy zwykłą fizyką MuJoCo (robot opada pod grawitacją,
    # bo siłowniki pozycyjne mają domyślny cel 0 — to normalne dla gołego modelu).
    step_fn = mujoco.mj_step if args.simulate else None

    server = viser.ViserServer(port=args.port)
    viewer = Viewer(model, data, step_fn=step_fn, server=server)
    print(
        f"\nViser działa — otwórz w przeglądarce:  http://localhost:{args.port}\n"
        f"(na zdalnym serwerze: tunel SSH  ->  "
        f"ssh -L {args.port}:localhost:{args.port} użytkownik@host)\n"
        "Ctrl+C kończy.",
        flush=True,
    )
    viewer.run()


if __name__ == "__main__":
    main()

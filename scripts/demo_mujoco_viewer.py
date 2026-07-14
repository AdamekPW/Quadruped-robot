"""Demo 2 — interaktywny podgląd MuJoCo (samo "czucie" symulatora).

Otwiera natywny viewer MuJoCo z modelem czworonoga. Bez sterowania robot po prostu
opada pod wpływem grawitacji — chodzi o to, żeby poznać interfejs symulatora:

  • lewy przycisk myszy  — obrót kamery
  • prawy przycisk       — przesuwanie kamery
  • scroll               — zoom
  • Ctrl + lewy/prawy    — chwytanie i ciągnięcie części robota (przyłóż siłę!)
  • spacja               — pauza / wznów symulację
  • podwójny klik        — wybór ciała; panele po lewej pokazują stawy, kontakty itd.

To najlepszy sposób, żeby zobaczyć "jak wygląda praca w środowisku MuJoCo".

Uruchomienie:
    .venv\\Scripts\\activate
    python scripts\\demo_mujoco_viewer.py
"""
import gymnasium as gym
import mujoco.viewer

# Wykorzystujemy model Ant dostarczany z Gymnasium (plik MJCF pod spodem).
env = gym.make("Ant-v5")
model = env.unwrapped.model  # mjModel — skompilowana definicja robota i sceny
data = env.unwrapped.data    # mjData  — bieżący stan symulacji

# launch() sam prowadzi symulację (mj_step) i obsługuje interakcję myszą/klawiaturą.
mujoco.viewer.launch(model, data)
env.close()

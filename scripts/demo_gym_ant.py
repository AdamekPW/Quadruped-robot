"""Demo 1 — podgląd środowiska przez Gymnasium (API RL + renderowanie MuJoCo).

Uruchamia środowisko `Ant-v5` (czworonóg na silniku MuJoCo) z LOSOWĄ polityką.
Robot będzie się ruszał chaotycznie i przewracał — to normalne, bo nic się jeszcze
nie nauczył. Cel demo: pokazać pętlę reset() -> step() oraz renderowanie na żywo.

Ant to tymczasowy zamiennik robopsa — ma 4 nogi i 8 sterowanych stawów.
Docelowo w to miejsce wejdzie własny model MJCF robopsa.

Uruchomienie:
    .venv\\Scripts\\activate
    python scripts\\demo_gym_ant.py

Okno zamknij krzyżykiem lub Ctrl+C w terminalu.
"""
import gymnasium as gym

# render_mode="human" -> otwiera interaktywne okno i renderuje w czasie rzeczywistym
env = gym.make("Ant-v5", render_mode="human")

print("observation_space:", env.observation_space.shape)  # 105 liczb: pozycje, prędkości...
print("action_space:", env.action_space.shape, "(momenty w 8 stawach, zakres -1..1)")

obs, info = env.reset(seed=0)
episode, ep_reward = 0, 0.0
try:
    while True:
        action = env.action_space.sample()          # losowa akcja
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += float(reward)
        if terminated or truncated:                 # upadek albo koniec epizodu
            episode += 1
            print(f"epizod {episode}: suma nagrody = {ep_reward:.1f}")
            obs, info = env.reset()
            ep_reward = 0.0
except KeyboardInterrupt:
    print("\nprzerwano")
finally:
    env.close()

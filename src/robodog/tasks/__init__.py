"""Zadania RL dla RoboDoga.

Import tego pakietu rejestruje zadania w rejestrze mjlab. mjlab ładuje go
automatycznie przez entry point grupy "mjlab.tasks" (patrz pyproject.toml).
"""

# Import podpakietu uruchamia jego __init__, który woła register_mjlab_task.
from robodog.tasks.velocity.config import silver_badger  # noqa: F401

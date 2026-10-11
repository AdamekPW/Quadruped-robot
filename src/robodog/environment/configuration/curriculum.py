""" Curriculum - stopniowe podkręcanie trudności w trakcie treningu """

from mjlab.managers.curriculum_manager import CurriculumTermCfg

from .. import mdp as robodog_mdp

# Zdarzenia objęte narastaniem siły.
_RANDOMIZED_EVENTS = (
    "encoder_bias",
    "foot_friction",
    "pd_gains",
    "joint_friction",
    "gravity",
    "push_robot",
    "pseudo_inertia",
    "joint_armature",
)

# Waga pojedynczego epizodu w średniej kroczącej `terrain_mastery`. 0.01 daje
# stałą czasową ~100 epizodów NA KOMÓRKĘ siatki; przy 2048 środowiskach kończy
# się ich ~50 na iterację, więc krzywa wygładza się w skali ~150 iteracji.
_MASTERY_SMOOTHING = 0.01

# Kroki liczone w krokach środowiska: iteracja PPO to 24 kroki (`num_steps_per_env`),
# więc randomizacja rusza po ~500 iteracjach i osiąga 100% po ~4000.
_STEPS_PER_ITERATION = 24
_START_ITERATION = 100_500
_END_ITERATION = 1004_000


def prepare_curriculum(curriculum: dict[str, CurriculumTermCfg]) -> None:
    """ Dokłada pomiar opanowania terenu i narastanie siły domain randomization """
    # Pomiar MUSI iść przed bazowym `terrain_levels`, bo tamten przy awansie
    # nadpisuje poziom, typ i origin resetowanych środowisk — po nim dystans i
    # przypisanie do komórki siatki byłyby liczone dla już zmienionego terenu.
    # Słownik przepisujemy w miejscu, bo `env_cfg` dostaje go przez referencję.
    existing = dict(curriculum)
    curriculum.clear()
    curriculum["terrain_mastery"] = CurriculumTermCfg(
        func=robodog_mdp.terrain_mastery,
        params={"smoothing": _MASTERY_SMOOTHING},
    )
    curriculum.update(existing)

    curriculum["domain_randomization"] = CurriculumTermCfg(
        func=robodog_mdp.dr_curriculum,
        params={
            "event_names": _RANDOMIZED_EVENTS,
            "start_step": _START_ITERATION * _STEPS_PER_ITERATION,
            "end_step": _END_ITERATION * _STEPS_PER_ITERATION,
            "action_name": "joint_pos",
        },
    )

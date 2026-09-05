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

# Kroki liczone w krokach środowiska: iteracja PPO to 24 kroki (`num_steps_per_env`),
# więc randomizacja rusza po ~500 iteracjach i osiąga 100% po ~4000.
_STEPS_PER_ITERATION = 24
_START_ITERATION = 500
_END_ITERATION = 4_000


def prepare_curriculum(curriculum: dict[str, CurriculumTermCfg]) -> None:
    """ Dokłada narastanie siły domain randomization """
    curriculum["domain_randomization"] = CurriculumTermCfg(
        func=robodog_mdp.dr_curriculum,
        params={
            "event_names": _RANDOMIZED_EVENTS,
            "start_step": _START_ITERATION * _STEPS_PER_ITERATION,
            "end_step": _END_ITERATION * _STEPS_PER_ITERATION,
            "action_name": "joint_pos",
        },
    )

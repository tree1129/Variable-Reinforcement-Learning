"""Independent Case 4-7 reinforcement-learning package."""

from .tasks import CASES, FACE_ORDER, SHAPE_ORDER, TaskSpec, build_prompt, get_task
from .reward import RewardConfig, RewardResult, benchmark_score, compute_reward

__all__ = [
    "CASES",
    "FACE_ORDER",
    "SHAPE_ORDER",
    "TaskSpec",
    "RewardConfig",
    "RewardResult",
    "benchmark_score",
    "build_prompt",
    "compute_reward",
    "get_task",
]

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Callable

import yaml

from .tasks import CASES, build_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "case4_7_rl.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "case4_7_sac"


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def validate() -> int:
    for case_id, task in CASES.items():
        print(f"Case {case_id}: {build_prompt(case_id)}")
        print(f"  targets={task.targets}; excluded={task.excluded}; placements={dict(task.placements)}")
    print("OK: Cases 4–7 task mappings validated")
    return 0


def _factory(spec: str) -> Callable[[], Any]:
    module_name, separator, attr = spec.partition(":")
    if not separator:
        raise ValueError("--backend-factory must be module:callable")
    result = getattr(importlib.import_module(module_name), attr)
    if not callable(result):
        raise TypeError(f"{spec} is not callable")
    return result


def train() -> int:
    parser = argparse.ArgumentParser(description="Train SAC/PPO for independent Cases 4–7 RL")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--backend", choices=("toy", "mujoco", "custom"), default="toy")
    parser.add_argument("--backend-factory", default="")
    parser.add_argument("--algorithm", choices=("sac", "ppo"), default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", type=Path, default=None, help="Warm-start from a saved SAC/PPO policy; replay buffer is intentionally rebuilt")
    parser.add_argument("--device", default=None, help="SB3 device, e.g. cuda or cpu (default: config device or auto)")
    parser.add_argument("--progress-bar", action="store_true", help="Enable SB3 rich/tqdm progress bar")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    algorithm = (args.algorithm or config.get("algorithm", "sac")).lower()
    seed = config.get("seed", 0) if args.seed is None else args.seed
    steps = args.steps or int(config.get("training", {}).get("total_timesteps", 300000))
    device = args.device or config.get("device", "auto")

    if args.dry_run:
        print(json.dumps({
            "algorithm": algorithm, "steps": steps, "device": device, "cases": {
                str(i): {"prompt": build_prompt(i), "targets": list(t.targets), "excluded": t.excluded, "hole_faces": dict(t.hole_faces)}
                for i, t in CASES.items()
            }, "note": "dry-run only; no policy was trained",
        }, ensure_ascii=False, indent=2))
        return 0

    try:
        from stable_baselines3 import PPO, SAC
    except ImportError as exc:
        raise SystemExit("Missing RL dependencies. Run: python3 -m pip install -e '.[rl]'") from exc
    if args.backend == "custom" and not args.backend_factory:
        parser.error("--backend custom requires --backend-factory module:callable")

    from .env import Case47Env
    from .reward import RewardConfig
    from .toy_backend import ToyBackend

    if args.backend == "toy":
        backend = ToyBackend(seed)
    elif args.backend == "mujoco":
        factory = args.backend_factory or "tree_reinforcement_learning.mujoco_backend:make_backend"
        backend = _factory(factory)()
    else:
        backend = _factory(args.backend_factory)()
    env = Case47Env(backend, max_episode_steps=int(config.get("max_episode_steps", 600)), reward_config=RewardConfig(**config.get("reward", {})), seed=seed)
    train_config = config.get("training", {})
    model_class = SAC if algorithm == "sac" else PPO
    common = {
        "learning_rate": train_config.get("learning_rate", 3e-4), "batch_size": train_config.get("batch_size", 256),
        "gamma": train_config.get("gamma", 0.99), "seed": seed, "verbose": 1,
        "device": device,
    }
    # TensorBoard is optional. Only enable the SB3 logger when the package is
    # installed, so a clean A100 venv can still run a smoke test immediately.
    try:
        import tensorboard  # noqa: F401
    except ImportError:
        pass
    else:
        common["tensorboard_log"] = str(args.output / "tensorboard")
    if algorithm == "sac":
        common.update({"buffer_size": train_config.get("buffer_size", 100000), "learning_starts": train_config.get("learning_starts", 5000), "tau": train_config.get("tau", 0.005), "ent_coef": train_config.get("ent_coef", "auto")})
    if args.resume is not None:
        if not args.resume.exists():
            parser.error(f"--resume checkpoint not found: {args.resume}")
        model = model_class.load(str(args.resume), env=env, device=device)
        # The previous round used a different shaped reward, so do not reuse its
        # replay buffer; warm-start only the policy/value weights.
        model.set_random_seed(seed)
    else:
        model = model_class("MlpPolicy", env, **common)
    args.output.mkdir(parents=True, exist_ok=True)
    model.learn(total_timesteps=steps, progress_bar=args.progress_bar)
    model.save(str(args.output / f"{algorithm}_case4_7"))
    env.close()
    print(f"Saved policy: {args.output / (algorithm + '_case4_7')}")
    return 0

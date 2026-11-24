"""T3 benchmark (walk with a base-velocity command while tracking a 6D TCP target), protocol v0.6.

Pure modules (importable without Isaac Lab): protocol, frames, kinematics, pool, goals, metrics, eval_set.
Isaac Lab terms: runtime; environment configs: env_cfg (training and evaluation).
"""

import gymnasium as gym

gym.register(
    id="RobotLab-Isaac-T3-WalkReach-Unitree-B2W-Z1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:T3WalkReachEnvCfg"},
)

gym.register(
    id="RobotLab-Isaac-T3-WalkReach-Eval-Unitree-B2W-Z1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:T3WalkReachEvalEnvCfg"},
)

"""T2 benchmark (reach low and high targets while standing), protocol v0.4.

Separate from the shipped RFM tasks and from T1. The env needs the frozen full-grid reachability map
(version t2-reachability-3): ReachTargetCommandCfg.reachability_path or env var T2_REACHABILITY_NPZ.
"""

import gymnasium as gym

gym.register(
    id="RobotLab-Isaac-T2-Reach-Unitree-B2W-Z1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:T2ReachEnvCfg"},
)

gym.register(
    id="RobotLab-Isaac-T2-Reach-Eval-Unitree-B2W-Z1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:T2ReachEvalEnvCfg"},
)

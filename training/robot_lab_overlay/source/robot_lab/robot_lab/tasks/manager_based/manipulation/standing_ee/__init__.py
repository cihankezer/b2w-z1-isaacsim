"""First thesis benchmark task; deliberately separate from the shipped RFM tasks."""

import gymnasium as gym

gym.register(
    id="RobotLab-Isaac-T1-Standing-Unitree-B2W-Z1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:StandingEnvCfg"},
)

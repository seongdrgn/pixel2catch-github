import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="pixel2catch",
    entry_point=f"{__name__}.pixel2catch:DynamicCatchEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pixel2catch:DynamicCatchEnvCfg",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:pixel2catch.yaml",
    },
)

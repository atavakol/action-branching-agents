from gym.envs.registration import register

# MuJoCO environments
register(id='Reacher3DOF-v0', entry_point='envs.mujoco.reacher_3dof:Reacher3DOFEnv', max_episode_steps=50)
register(id='Reacher4DOF-v0', entry_point='envs.mujoco.reacher_4dof:Reacher4DOFEnv', max_episode_steps=60)
register(id='Reacher5DOF-v0', entry_point='envs.mujoco.reacher_5dof:Reacher5DOFEnv', max_episode_steps=70)
register(id='Reacher6DOF-v0', entry_point='envs.mujoco.reacher_6dof:Reacher6DOFEnv', max_episode_steps=80)
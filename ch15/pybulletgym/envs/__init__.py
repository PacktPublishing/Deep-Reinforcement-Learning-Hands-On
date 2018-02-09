from gym.envs.registration import register

register(
	id='InvertedPendulumPyBulletEnv-v0',
	entry_point='pybulletgym.envs.gym_pendulum_envs:InvertedPendulumBulletEnv',
	max_episode_steps=1000,
	reward_threshold=950.0,
	)

register(
	id='InvertedDoublePendulumPyBulletEnv-v0',
	entry_point='pybulletgym.envs.gym_pendulum_envs:InvertedDoublePendulumBulletEnv',
	max_episode_steps=1000,
	reward_threshold=9100.0,
	)

register(
	id='InvertedPendulumSwingupPyBulletEnv-v0',
	entry_point='pybulletgym.envs.gym_pendulum_envs:InvertedPendulumSwingupBulletEnv',
	max_episode_steps=1000,
	reward_threshold=800.0,
	)

register(
	id='ReacherPyBulletEnv-v0',
	entry_point='pybulletgym.envs.gym_manipulator_envs:ReacherBulletEnv',
	max_episode_steps=150,
	reward_threshold=18.0,
	)

register(
	id='PusherPyBulletEnv-v0',
	entry_point='pybulletgym.envs.gym_manipulator_envs:PusherBulletEnv',
	max_episode_steps=150,
	reward_threshold=18.0,
)

register(
	id='ThrowerPyBulletEnv-v0',
	entry_point='pybulletgym.envs.gym_manipulator_envs:ThrowerBulletEnv',
	max_episode_steps=100,
	reward_threshold=18.0,
)

register(
	id='StrikerPyBulletEnv-v0',
	entry_point='pybulletgym.envs.gym_manipulator_envs:StrikerBulletEnv',
	max_episode_steps=100,
	reward_threshold=18.0,
)

register(
	id='Walker2DPyBulletEnv-v0',
	entry_point='pybulletgym.envs.gym_locomotion_envs:Walker2DBulletEnv',
	max_episode_steps=1000,
	reward_threshold=2500.0
	)
register(
	id='HalfCheetahPyBulletEnv-v0',
	entry_point='pybulletgym.envs.gym_locomotion_envs:HalfCheetahBulletEnv',
	max_episode_steps=1000,
	reward_threshold=3000.0
	)

register(
	id='AntPyBulletEnv-v0',
	entry_point='pybulletgym.envs.gym_locomotion_envs:AntBulletEnv',
	max_episode_steps=1000,
	reward_threshold=2500.0
	)

register(
	id='HopperPyBulletEnv-v0',
	entry_point='pybulletgym.envs.gym_locomotion_envs:HopperBulletEnv',
	max_episode_steps=1000,
	reward_threshold=2500.0
	)

register(
	id='HumanoidPyBulletEnv-v0',
	entry_point='pybulletgym.envs.gym_locomotion_envs:HumanoidBulletEnv',
	max_episode_steps=1000
	)

register(
	id='HumanoidFlagrunPyBulletEnv-v0',
	entry_point='pybulletgym.envs.gym_locomotion_envs:HumanoidFlagrunBulletEnv',
	max_episode_steps=1000,
	reward_threshold=2000.0
	)

register(
	id='HumanoidFlagrunHarderPyBulletEnv-v0',
	entry_point='pybulletgym.envs.gym_locomotion_envs:HumanoidFlagrunHarderBulletEnv',
	max_episode_steps=1000
	)

register(
	id='AtlasPyBulletEnv-v0',
	entry_point='pybulletgym.envs.gym_locomotion_envs:AtlasBulletEnv',
	max_episode_steps=1000
	)

def getList():
	btenvs = ['- ' + spec.id for spec in gym.pgym.envs.registry.all() if spec.id.find('Bullet')>=0]
	return btenvs

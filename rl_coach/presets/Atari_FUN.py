import numpy as np

from rl_coach.agents.fun_agent import FUNAgentParameters, FUNJob
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters
from rl_coach.core_types import EnvironmentEpisodes, EnvironmentSteps, TrainingSteps
from rl_coach.environments.gym_environment import GymVectorEnvironment
from rl_coach.exploration_policies.e_greedy import EGreedyParameters
from rl_coach.exploration_policies.ou_process import OUProcessParameters
from rl_coach.graph_managers.graph_manager import ScheduleParameters
from rl_coach.graph_managers.hac_graph_manager import HACGraphManager
from rl_coach.memories.memory import MemoryGranularity
from rl_coach.schedules import ConstantSchedule
from rl_coach.spaces import GoalsSpace, ReachingGoal


### Manager NetworK: DiLSTM, with 256 hidden units

# SET THE EMBEDDER TO MEDIUM and the following:
# Followed by fully connected 256 hidden layer
# Activcation is relu for all layers

## state_space = Manager state space formulate the goal using a fully connected layer followed by relu activation
# goal t i s generated using manager's rnn network
# embedding vector w is set to k = 16
# In the paper, two actor critic network are created and trained separately

# INPUT is defined such as
# fpercept: a CNN followed by fully connected layer.
# CNN first layer: 16 8x8 filter of stride 4, followed by 32 4x4 filters of stride 2
# worker_agent_params.input_embedders_parameters = {'observation': InputEmbedderParameters(), 'desired_goal' }
# should the manager output go through another softmax layer?
# defaults for worker are fine
#### Worker Network: Simple LSTM, with 256 hidden units



####################
# Graph Scheduling #
####################
schedule_params = ScheduleParameters()
schedule_params.improve_steps = EnvironmentEpisodes(40 * 4 * 64)  # 40 epochs
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(4 * 64)  # 4 small batches of 64 episodes
schedule_params.evaluation_steps = EnvironmentEpisodes(64)
schedule_params.heatup_steps = EnvironmentSteps(0)

#########
# Agent #
#########
manager_agent_params = FUNAgentParameters(FUNJob.Manager)
worker_agent_params = FUNAgentParameters(FUNJob.Worker)

# TODO once Dilated lstm is implemeneted, replace the middleware layer with dilated lstm over lstm

goals_space = GoalsSpace('achieved_goal',
                         ReachingGoal(distance_from_goal_threshold=np.array([0.09, 0.09, 0.09])),
                         distance_metric=GoalsSpace.DistanceMetric.Cosine)  # use cos distance as paper3

manager_agent_params.memory.goals_space = goals_space
worker_agent_params.algorithm.in_action_space = goals_space
worker_agent_params.memory.goals_space = goals_space

# Inspired by the other pendulum env, those hyperparameters seems to work best
worker_agent_params.algorithm.num_consecutive_playing_steps = EnvironmentEpisodes(16 * 25)
manager_agent_params.algorithm.num_consecutive_playing_steps = EnvironmentEpisodes(32)

agents_params = [manager_agent_params, worker_agent_params]

###############
# Environment #
###############
# env_params = Atari(level=SingleLevelSelection(atari_deterministic_v4))
time_limit = 1000

env_params = GymVectorEnvironment(level="rl_coach.environments.mujoco.pendulum_with_goals:PendulumWithGoals")
env_params.additional_simulator_parameters = {"time_limit": time_limit,
                                              "random_goals_instead_of_standing_goal": False,
                                              "polar_coordinates": False,
                                              "goal_reaching_thresholds": np.array([0.09, 0.09, 0.09])}
env_params.frame_skip = 10
env_params.custom_reward_threshold = -time_limit + 1

########
# Test #
########
preset_validation_params = PresetValidationParameters()
preset_validation_params.trace_test_levels = ['breakout', 'ms_pacman', 'space_invaders', 'PendulumWithGoals']

# use a hierarchical graph manager
graph_manager = HACGraphManager(agents_params=agents_params, env_params=env_params,
                                schedule_params=schedule_params,
                                vis_params=VisualizationParameters(native_rendering=False),
                                preset_validation_params=preset_validation_params,
                                consecutive_steps_to_run_non_top_levels=EnvironmentSteps(40))

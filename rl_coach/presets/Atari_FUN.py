import numpy as np

from rl_coach.agents.fun_agent import FUNAgentParameters
from rl_coach.base_parameters import VisualizationParameters, PresetValidationParameters
from rl_coach.core_types import TrainingSteps, EnvironmentEpisodes, EnvironmentSteps
from rl_coach.environments.environment import SingleLevelSelection
from rl_coach.environments.gym_environment import Atari, atari_deterministic_v4, GymVectorEnvironment
from rl_coach.graph_managers.graph_manager import ScheduleParameters
####################
# Graph Scheduling #
####################
from rl_coach.graph_managers.hrl_graph_manager import HRLGraphManager
from rl_coach.spaces import GoalsSpace, ReachingGoal

schedule_params = ScheduleParameters()
schedule_params.improve_steps = TrainingSteps(10000000000)
schedule_params.steps_between_evaluation_periods = EnvironmentEpisodes(100)
schedule_params.evaluation_steps = EnvironmentEpisodes(3)
schedule_params.heatup_steps = EnvironmentSteps(0)

#########
# Agent #
#########
# In the paper, two actor critic network are created and trained separately

# INPUT is defined such as
# fpercept: a CNN followed by fully connected layer.
# CNN first layer: 16 8x8 filter of stride 4, followed by 32 4x4 filters of stride 2


goals_space = GoalsSpace('achieved_goal',
                         ReachingGoal(default_reward=-1, goal_reaching_reward=0,
                                      distance_from_goal_threshold=np.array([0.075, 0.75])),
                         distance_metric=GoalsSpace.DistanceMetric.Cosine)  # use cos distance as paper

### Manager NetworK: DiLSTM, with 256 hidden units
manager_agent_params = FUNAgentParameters()
# Manager learns on a slower temporal
manager_agent_params.network_wrappers['critic'].learning_rate = 0.001
manager_agent_params.network_wrappers['actor'].learning_rate = 0.001

# TODO once Dilated lstm is implemeneted, replace the middleware layer with dilated lstm over lstm

worker_agent_params = FUNAgentParameters()
worker_agent_params.algorithm.in_action_space = goals_space
worker_agent_params.memory.goals_space = goals_space

# defaults for worker are fine
#### Worker Network: Simple LSTM, with 256 hidden units

agents_params = [manager_agent_params, worker_agent_params]

# SET THE EMBEDDER TO MEDIUM and the following:
# Followed by fully connected 256 hidden layer
# Activcation is relu for all layers

## state_space = Manager state space formulate the goal using a fully connected layer followed by relu activation
# goal t i s generated using manager's rnn network
# embedding vector w is set to k = 16

###############
# Environment #
###############
# env_params = Atari(level=SingleLevelSelection(atari_deterministic_v4))
env_params = GymVectorEnvironment(level="rl_coach.environments.mujoco.pendulum_with_goals:PendulumWithGoals")

########
# Test #
########
preset_validation_params = PresetValidationParameters()
preset_validation_params.trace_test_levels = ['breakout', 'ms_pacman', 'space_invaders', 'PendulumWithGoals'] # -lvl ms_pacman

# use a hierarchical graph manager
graph_manager = HRLGraphManager(agents_params=agents_params, env_params=env_params,
                                schedule_params=schedule_params, vis_params=VisualizationParameters(),
                                preset_validation_params=preset_validation_params,
                                consecutive_steps_to_run_each_level=EnvironmentSteps(50))

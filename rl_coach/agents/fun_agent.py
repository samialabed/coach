#
# Copyright (c) 2017 Intel Corporation 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from collections import OrderedDict
from enum import Enum
from typing import Union

from rl_coach.agents.composite_agent import CompositeAgent
from rl_coach.agents.ddpg_agent import DDPGAlgorithmParameters, DDPGAgent, DDPGCriticNetworkParameters, \
    DDPGActorNetworkParameters
from rl_coach.architectures.embedder_parameters import InputEmbedderParameters
from rl_coach.architectures.head_parameters import FUNActorHeadParameters
from rl_coach.architectures.layers import Dense
from rl_coach.architectures.middleware_parameters import LSTMMiddlewareParameters
from rl_coach.base_parameters import AgentParameters, MiddlewareScheme, EmbedderScheme
from rl_coach.core_types import EnvironmentEpisodes, TrainingSteps
from rl_coach.exploration_policies.e_greedy import EGreedyParameters
from rl_coach.level_manager import LevelManager
from rl_coach.memories.episodic import EpisodicHRLHindsightExperienceReplayParameters, \
    EpisodicHindsightExperienceReplayParameters
from rl_coach.memories.episodic.episodic_hindsight_experience_replay import HindsightGoalSelectionMethod
from rl_coach.spaces import SpacesDefinition


class FUNJob(Enum):
    Worker = 'worker'
    Manager = 'manager'


class FUNCriticNetworkParameters(DDPGCriticNetworkParameters):
    def __init__(self):
        super(FUNCriticNetworkParameters, self).__init__()
        self.middleware_parameters = LSTMMiddlewareParameters(scheme=[Dense(64)] * 3,
                                                              number_of_lstm_cells=256)
        # Shallow is conv2d with similar layer structure as the paper
        self.input_embedders_parameters = {'observation': InputEmbedderParameters(scheme=EmbedderScheme.Shallow),
                                           'action': InputEmbedderParameters(scheme=EmbedderScheme.Empty),
                                           'desired_goal': InputEmbedderParameters(scheme=EmbedderScheme.Empty)}
        self.optimizer_type = 'Adam'
        self.batch_size = 4096
        # Consider enabling ActionInfo interensic reward


class FUNActorNetworkParameters(DDPGActorNetworkParameters):
    def __init__(self):
        super(FUNActorNetworkParameters, self).__init__()
        self.middleware_parameters = LSTMMiddlewareParameters(scheme=[Dense(64)] * 3,
                                                              number_of_lstm_cells=256)
        self.input_embedders_parameters = {'observation': InputEmbedderParameters(scheme=EmbedderScheme.Shallow),
                                           'desired_goal': InputEmbedderParameters(scheme=EmbedderScheme.Empty)}
        self.heads_parameters = [FUNActorHeadParameters()]
        self.optimizer_type = 'Adam'
        self.batch_size = 4096


class FUNAlgorithmParameters(DDPGAlgorithmParameters):
    def __init__(self):
        super(FUNAlgorithmParameters, self).__init__()
        self.num_consecutive_training_steps = 40
        self.num_steps_between_copying_online_weights_to_target = TrainingSteps(40)
        self.num_consecutive_playing_steps = EnvironmentEpisodes(32)


class FUNAgentParameters(AgentParameters):
    def __init__(self, feudal_type: FUNJob = FUNJob.Worker):
        actor = FUNActorNetworkParameters()
        critic = FUNCriticNetworkParameters()
        # Manager specific setup
        if feudal_type == FUNJob.Manager:
            # Manager will have a different actor head in the future that simulate eq 6 from paper
            memory = EpisodicHRLHindsightExperienceReplayParameters()
            memory.hindsight_transitions_per_regular_transition = 3  # Slower sampling rate
            actor.learning_rate = 0.0001
            critic.learning_rate = 0.0001
        # Worker specific setup
        elif feudal_type == FUNJob.Worker:
            memory = EpisodicHindsightExperienceReplayParameters()
            memory.hindsight_transitions_per_regular_transition = 5  # Faster sampling rate
            actor.learning_rate = 0.001
            critic.learning_rate = 0.001
        else:
            raise ValueError("Select either Manager or Worker for the feudal network")

        # common parameter setup
        memory.hindsight_goal_selection_method = HindsightGoalSelectionMethod.Future

        super(FUNAgentParameters, self).__init__(algorithm=FUNAlgorithmParameters(),
                                                 exploration=EGreedyParameters(),
                                                 memory=memory,
                                                 networks=OrderedDict([("critic", critic),
                                                                       ("actor", actor)]))

    @property
    def path(self):
        return 'rl_coach.agents.fun_agent:FUNAgent'


class FUNAgent(DDPGAgent):
    """
    Implementation of Feudal Network defined in [1].

    TODO - document parameters
    Using DDPG proved better results than A-C in complex goal setup env (A-C Doesn't work in multiple goal env)
    [1] FeUdal Networks for Hierarchical Reinforcement Learning - https://arxiv.org/abs/1703.01161
    """

    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent'] = None):
        super(FUNAgent, self).__init__(agent_parameters, parent)

    def learn_from_batch(self, batch):
        # TODO once implemented dilated-lstm include a softmax on workers batch (formula 10)
        super(FUNAgent, self).learn_from_batch(batch)

    def train(self):
        # each network is train disjointly, nothing special done here
        super(FUNAgent, self).train()

    def choose_action(self, curr_state):
        return super(FUNAgent, self).choose_action(curr_state)

    def update_transition_before_adding_to_replay_buffer(self, transition):
        # workers are lowest point in a feudal system - set goal based on managers output
        # use intrinsic reward for worker
        if self.ap.is_a_lowest_level_agent:
            # Current_hrl_goal is set based on the manager action automatically at the level manager
            transition.state['desired_goal'] = self.current_hrl_goal
            transition.next_state['desired_goal'] = self.current_hrl_goal
            self.distance_from_goal.add_sample(self.spaces.goal.distance_from_goal(
                self.current_hrl_goal, transition.next_state))
            goal_reward, sub_goal_reached = self.spaces.goal.get_reward_for_goal_and_state(
                self.current_hrl_goal, transition.next_state)
            # Reward worker for achieving goal set by manager
            transition.reward = goal_reward
            transition.game_over = transition.game_over or sub_goal_reached

        # Assume highest level is the manager set goal reward
        if self.ap.is_a_highest_level_agent:
            subgoal_reward, sub_goal_reached = self.spaces.goal.get_reward_for_goal_and_state(transition.action,
                                                                                              transition.next_state)
            if not sub_goal_reached:
                transition.reward = -subgoal_reward

        # TODO - should we support middle level managers?

        return transition

    def set_environment_parameters(self, spaces: SpacesDefinition):
        super(FUNAgent, self).set_environment_parameters(spaces)

        # let the manager set the goal for workers based on how far they achieved the desired goal
        if self.ap.is_a_highest_level_agent:
            self.spaces.goal = self.spaces.action
            self.spaces.goal.set_target_space(self.spaces.state[self.spaces.goal.goal_name])

        if not self.ap.is_a_highest_level_agent:
            self.spaces.reward.reward_success_threshold = self.spaces.goal.reward_type.goal_reaching_reward

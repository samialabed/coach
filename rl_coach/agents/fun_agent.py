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
from typing import Union
import numpy as np

from rl_coach.agents.actor_critic_agent import ActorCriticAgent
from rl_coach.agents.composite_agent import CompositeAgent
from rl_coach.architectures.embedder_parameters import InputEmbedderParameters
from rl_coach.architectures.head_parameters import FUNActorHeadParameters, VHeadParameters
from rl_coach.architectures.layers import Dense
from rl_coach.architectures.middleware_parameters import LSTMMiddlewareParameters
from rl_coach.base_parameters import NetworkParameters, AlgorithmParameters, \
    AgentParameters, EmbedderScheme
from rl_coach.core_types import EnvironmentSteps, ActionInfo
from rl_coach.exploration_policies.e_greedy import EGreedyParameters
from rl_coach.level_manager import LevelManager
from rl_coach.memories.episodic import EpisodicHRLHindsightExperienceReplayParameters
from rl_coach.spaces import SpacesDefinition


class FUNCriticNetworkParameters(NetworkParameters):
    def __init__(self):
        super(FUNCriticNetworkParameters, self).__init__()
        self.input_embedders_parameters = {'observation': InputEmbedderParameters(batchnorm=True),
                                           'action': InputEmbedderParameters(scheme=EmbedderScheme.Shallow)}
        self.middleware_parameters = LSTMMiddlewareParameters(scheme=[Dense(256)])
        self.heads_parameters = [VHeadParameters()]
        self.optimizer_type = 'Adam'
        self.batch_size = 4096
        self.async_training = False
        self.learning_rate = 0.001
        self.create_target_network = True
        self.shared_optimizer = True
        self.scale_down_gradients_by_number_of_workers_for_sync_training = False


class FUNActorNetworkParameters(NetworkParameters):
    def __init__(self):
        super(FUNActorNetworkParameters, self).__init__()
        self.input_embedders_parameters = {'observation': InputEmbedderParameters(batchnorm=True)}
        self.middleware_parameters = LSTMMiddlewareParameters(scheme=[Dense(256)])
        self.heads_parameters = [FUNActorHeadParameters()]
        self.optimizer_type = 'Adam'
        self.batch_size = 4096
        self.async_training = False
        self.learning_rate = 0.001
        self.create_target_network = True
        self.shared_optimizer = True
        self.scale_down_gradients_by_number_of_workers_for_sync_training = False


class FUNAlgorithmParameters(AlgorithmParameters):
    def __init__(self):
        super(FUNAlgorithmParameters, self).__init__()
        self.num_steps_between_copying_online_weights_to_target = EnvironmentSteps(1)
        self.rate_for_copying_weights_to_target = 0.001
        self.num_consecutive_playing_steps = EnvironmentSteps(1)


class FUNAgentParameters(AgentParameters):
    def __init__(self):
        super(FUNAgentParameters, self).__init__(algorithm=FUNAlgorithmParameters(),
                                                 exploration=EGreedyParameters(),
                                                 memory=EpisodicHRLHindsightExperienceReplayParameters(),
                                                 networks=OrderedDict([
                                                     ("actor", FUNActorNetworkParameters()),
                                                     ("critic", FUNCriticNetworkParameters()),
                                                 ]))

    @property
    def path(self):
        return 'rl_coach.agents.fun_agent:FUNAgent'


class FUNAgent(ActorCriticAgent):
    """
    Implementation of Feudal Network defined in [1].

    TODO - document parameters

    [1] FeUdal Networks for Hierarchical Reinforcement Learning - https://arxiv.org/abs/1703.01161
    """

    def __init__(self, agent_parameters, parent: Union['LevelManager', 'CompositeAgent'] = None):
        super(FUNAgent, self).__init__(agent_parameters, parent)

    def learn_from_batch(self, batch):
        # TODO once implemented dilated-lstm include a softmax on workers batch (formula 10)
        super(FUNAgent).learn_from_batch(batch)

    def train(self):
        super(FUNAgent, self).train()

    def choose_action(self, curr_state):
        tf_input_state = self.prepare_batch_for_inference(curr_state, 'actor')

        actor_network = self.networks['actor'].online_network

        action_values = actor_network.predict(tf_input_state).squeeze()

        action = self.exploration_policy.get_action(action_values)


        # get q value
        tf_input_state = self.prepare_batch_for_inference(curr_state, 'critic')
        action_batch = np.expand_dims(action, 0)
        if type(action) != np.ndarray:
            action_batch = np.array([[action]])
        tf_input_state['action'] = action_batch
        q_value = self.networks['critic'].online_network.predict(tf_input_state)[0]

        action_info = ActionInfo(action=action,
                                 action_value=q_value)

        return action_info

        # Use out of the shelf ActorCritic algorithm
        # The paper isn't clear on the manager implementation, for now go with defaults

    def update_transition_before_adding_to_replay_buffer(self, transition):
        # workers are lowest point in a feudal system - set goal based on managers output
        # use intrinsic reward for worker
        if self.ap.is_a_lowest_level_agent:
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
            _, sub_goal_reached = self.spaces.goal.get_reward_for_goal_and_state(
                transition.action, transition.next_state)

            sub_goal_is_missed = not sub_goal_reached

            if sub_goal_is_missed:
                transition.reward = -self.ap.algorithm.time_limit

        # TODO - should we support middle level managers?

        return transition

    def set_environment_parameters(self, spaces: SpacesDefinition):
        super(FUNAgent, self).set_environment_parameters(spaces)

        if self.ap.is_a_highest_level_agent:
            self.spaces.goal = self.spaces.action
            self.spaces.goal.set_target_space(self.spaces.state[self.spaces.goal.goal_name])

        if not self.ap.is_a_highest_level_agent:
            self.spaces.reward.reward_success_threshold = self.spaces.goal.reward_type.goal_reaching_reward

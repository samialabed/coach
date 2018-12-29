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


import tensorflow as tf

from rl_coach.architectures.tensorflow_components.layers import Dense
from rl_coach.architectures.tensorflow_components.middlewares.middleware import Middleware
from rl_coach.base_parameters import MiddlewareScheme
from rl_coach.core_types import Middleware_LSTM_Embedding


# TODO implement this
class DilatedLSTMMiddleware(Middleware):
    def __init__(self, activation_function=tf.nn.relu, number_of_lstm_cells: int = 256,
                 scheme: MiddlewareScheme = MiddlewareScheme.Medium,
                 batchnorm: bool = False, dropout_rate: float = 0.0,
                 name="middleware_lstm_embedder", dense_layer=Dense, is_training=False):
        super(DilatedLSTMMiddleware, self).__init__(activation_function=activation_function, batchnorm=batchnorm,
                                                    dropout_rate=dropout_rate, scheme=scheme, name=name,
                                                    dense_layer=dense_layer,
                                                    is_training=is_training)
        self.return_type = Middleware_LSTM_Embedding
        self.number_of_lstm_cells = number_of_lstm_cells
        self.layers = []

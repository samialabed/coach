Exploration Policies
====================

Exploration policies are a component that allow the agent to tradeoff exploration and exploitation according to a
predefined policy. This is one of the most important aspects of reinforcement learning agents, and can require some
tuning to get it right. Coach supports several pre-defined exploration policies, and it can be easily extended with
custom policies. Note that not all exploration policies are expected to work for both discrete and continuous action
spaces.

.. role:: green
.. role:: red

+----------------------+-----------------------+------------------+
| Exploration Policy   | Discrete Action Space | Box Action Space |
+======================+=======================+==================+
| AdditiveNoise        | :red:`X`              | :green:`V`       |
+----------------------+-----------------------+------------------+
| Boltzmann            | :green:`V`            | :red:`X`         |
+----------------------+-----------------------+------------------+
| Bootstrapped         | :green:`V`            | :red:`X`         |
+----------------------+-----------------------+------------------+
| Categorical          | :green:`V`            | :red:`X`         |
+----------------------+-----------------------+------------------+
| ContinuousEntropy    | :red:`X`              | :green:`V`       |
+----------------------+-----------------------+------------------+
| EGreedy              | :green:`V`            | :green:`V`       |
+----------------------+-----------------------+------------------+
| Greedy               | :green:`V`            | :green:`V`       |
+----------------------+-----------------------+------------------+
| OUProcess            | :red:`X`              | :green:`V`       |
+----------------------+-----------------------+------------------+
| ParameterNoise       | :green:`V`            | :green:`V`       |
+----------------------+-----------------------+------------------+
| TruncatedNormal      | :red:`X`              | :green:`V`       |
+----------------------+-----------------------+------------------+
| UCB                  | :green:`V`            | :red:`X`         |
+----------------------+-----------------------+------------------+

ExplorationPolicy
-----------------
.. autoclass:: rl_coach.exploration_policies.exploration_policy.ExplorationPolicy
   :members:
   :inherited-members:

AdditiveNoise
-------------
.. autoclass:: rl_coach.exploration_policies.additive_noise.AdditiveNoise

Boltzmann
---------
.. autoclass:: rl_coach.exploration_policies.boltzmann.Boltzmann

Bootstrapped
------------
.. autoclass:: rl_coach.exploration_policies.bootstrapped.Bootstrapped

Categorical
-----------
.. autoclass:: rl_coach.exploration_policies.categorical.Categorical

ContinuousEntropy
-----------------
.. autoclass:: rl_coach.exploration_policies.continuous_entropy.ContinuousEntropy

EGreedy
-------
.. autoclass:: rl_coach.exploration_policies.e_greedy.EGreedy

Greedy
------
.. autoclass:: rl_coach.exploration_policies.greedy.Greedy

OUProcess
---------
.. autoclass:: rl_coach.exploration_policies.ou_process.OUProcess

ParameterNoise
--------------
.. autoclass:: rl_coach.exploration_policies.parameter_noise.ParameterNoise

TruncatedNormal
---------------
.. autoclass:: rl_coach.exploration_policies.truncated_normal.TruncatedNormal

UCB
---
.. autoclass:: rl_coach.exploration_policies.ucb.UCB
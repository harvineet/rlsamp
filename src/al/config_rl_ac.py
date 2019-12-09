"""
Adapted from code by Ian Osband
https://github.com/iosband/ts_tutorial/

Specify the jobs to run via config file.

A simple experiment comparing policy gradient REINFORCE
and selective sampling BBQ algorithm.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import argparse

from base.config_lib import Config
from base.experiment import BaseExperiment
from al.env_al import ContextualBanditFunctionalContext
from al.agent_al import SelectiveSampleBBQ
from al.agent_rl import PolicyGradientREINFORCE
from al.agent_rl import PolicyGradientActorCritic
from al.agent_al import UniformRandom
from al.env_al import normal_iid, sinewave, linear_classifier
from al.policy import PolicyNNActorCritic
from al.policy import PolicyNN

def get_config():
  """Generates the config for the experiment."""
  name = 'rl_reinf_ac_bbq'
  true_theta = [1.0, 0.5]
  kappa_1 = 0.6
  kappa_2 = 0.8
  args_1 = argparse.Namespace() # From https://stackoverflow.com/questions/16878315/what-is-the-right-way-to-treat-python-argparse-namespace-as-a-dictionary
  args_1.n_feat = len(true_theta)
  args_1.optim = 'sgd'
  args_1.learn_rate = 1e-2
  args_1.momentum = 0.9 # only for SGD
  args_1.gamma = 1 # discount factor
  args_1.sample_cost = 2 # 0<=cost
  args_1.in_dim = 2 # input dim of policy
  args_1.n_act = 2 # num actions

  args_2 = argparse.Namespace() # From https://stackoverflow.com/questions/16878315/what-is-the-right-way-to-treat-python-argparse-namespace-as-a-dictionary
  args_2.n_feat = len(true_theta)
  args_2.optim = 'adam' #'sgd'
  args_2.learn_rate = 5e-3
  args_2.momentum = 0.9 # only for SGD
  args_2.gamma = 1 # discount factor
  args_2.sample_cost = 2 # 0 <= cost
  args_2.in_dim = 2 # input dim of policy
  args_2.n_act = 2 # num actions

  agents = collections.OrderedDict(
      [('bbq_k6',
        functools.partial(SelectiveSampleBBQ, args_1.n_feat, kappa_1)),
       ('bbq_k8',
        functools.partial(SelectiveSampleBBQ, args_1.n_feat, kappa_2)),
       ('reinf', functools.partial(PolicyGradientREINFORCE, PolicyNN, args_1)),
       ('ac', functools.partial(PolicyGradientActorCritic, PolicyNNActorCritic, args_2))]
  )
  environments = collections.OrderedDict(
      [('env', functools.partial(ContextualBanditFunctionalContext, normal_iid, true_theta, linear_classifier))]
  )
  experiments = collections.OrderedDict(
      [(name, BaseExperiment)]
  )
  n_steps = 200
  n_seeds = 10000
  config = Config(name, agents, environments, experiments, n_steps, n_seeds)
  return config
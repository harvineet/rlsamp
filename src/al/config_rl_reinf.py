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
from al.agent_al import UniformRandom
from al.env_al import normal_iid, sinewave, linear_classifier
from al.policy import PolicyNN

def get_config():
  """Generates the config for the experiment."""
  name = 'rl_bbq'
  true_theta = [1.0, 0.5]
  kappa_1 = 0.6
  kappa_2 = 0.8
  args = argparse.Namespace() # From https://stackoverflow.com/questions/16878315/what-is-the-right-way-to-treat-python-argparse-namespace-as-a-dictionary
  args.n_feat = len(true_theta)
  args.optim = 'sgd'
  args.learn_rate = 1e-2
  args.momentum = 0.9 # only for SGD
  args.gamma = 1 # discount factor
  args.sample_cost = 2 # 0<=cost
  args.in_dim = 2 # input dim of policy
  args.n_act = 2 # num actions
  agents = collections.OrderedDict(
      [('bbq_k6',
        functools.partial(SelectiveSampleBBQ, args.n_feat, kappa_1)),
       ('bbq_k8',
        functools.partial(SelectiveSampleBBQ, args.n_feat, kappa_2)),
       ('reinf', functools.partial(PolicyGradientREINFORCE, PolicyNN, args))]
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
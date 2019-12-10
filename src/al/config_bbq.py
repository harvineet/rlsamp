"""
Adapted from code by Ian Osband
https://github.com/iosband/ts_tutorial/

Specify the jobs to run via config file.

A simple experiment comparing selective sampling BBQ algorithm 
varying the query threshold. Higher threshold means more queries
and faster learning.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools

import numpy as np

from base.config_lib import Config
from base.experiment import BaseExperiment
from al.env_al import ContextualBanditFunctionalContext
from al.agent_al import SelectiveSampleBBQ
from al.agent_al import UniformRandom
from al.env_al import normal_iid, uniform_iid, sinewave, linear_classifier

def get_config():
  """Generates the config for the experiment."""
  name = 'bbq'
  true_theta = [1.0, 0.5]
#   unit_circle_angle = np.random.uniform(0, 2*np.pi) # sample 100-dim theta with first two non-zero and l2-norm theta=1
#   true_theta = [np.cos(unit_circle_angle), np.sin(unit_circle_angle)] + [0]*98
  kappa_1 = 0.25
  kappa_2 = 0.3
  n_feat = len(true_theta)
  agents = collections.OrderedDict(
      [('bbq_k25',
        functools.partial(SelectiveSampleBBQ, n_feat, kappa_1)),
       ('bbq_k3', functools.partial(SelectiveSampleBBQ, n_feat, kappa_2))]
  )
  environments = collections.OrderedDict(
      [('env', functools.partial(ContextualBanditFunctionalContext, normal_iid, true_theta, linear_classifier))]
    #   [('env', functools.partial(ContextualBanditFunctionalContext, uniform_iid, true_theta, linear_classifier))]
  )
  experiments = collections.OrderedDict(
      [(name, BaseExperiment)]
  )
  n_steps = 1000
  n_seeds = 10000
  config = Config(name, agents, environments, experiments, n_steps, n_seeds)
  return config
"""
Adapted from code by Ian Osband
https://github.com/iosband/ts_tutorial/

Specify the jobs to run via config file.

A simple experiment comparing Thompson sampling to greedy algorithm. Finite
armed bandit with 3 arms. Greedy algorithm premature and suboptimal
exploitation.
See Figure 3 from https://arxiv.org/abs/1707.02038
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools

from base.config_lib import Config
from base.experiment import BaseExperiment
from al.env_al import ContextualBanditFunctionalContext
from al.agent_al import SelectiveSampleBBQ
from al.agent_al import UniformRandom
from al.env_al import normal_iid, sinewave, linear_classifier

def get_config():
  """Generates the config for the experiment."""
  name = 'bbq'
  true_theta = [1.0, 0.5]
  kappa_1 = 0.8
  kappa_2 = 0.5
  n_feat = len(true_theta)
  agents = collections.OrderedDict(
      [('bbq_k1',
        functools.partial(SelectiveSampleBBQ, n_feat, kappa_1)),
       ('bbq_k2', functools.partial(SelectiveSampleBBQ, n_feat, kappa_2))]
  )
  environments = collections.OrderedDict(
      [('env', functools.partial(ContextualBanditFunctionalContext, normal_iid, true_theta, linear_classifier))]
  )
  experiments = collections.OrderedDict(
      [(name, BaseExperiment)]
  )
  n_steps = 1000
  n_seeds = 10000
  config = Config(name, agents, environments, experiments, n_steps, n_seeds)
  return config
"""
Adapted from code by Ian Osband
https://github.com/iosband/ts_tutorial/

Finite armed bandit environments."""

from __future__ import division
from __future__ import print_function

import numpy as np

from base.environment import Environment

##############################################################################

def sinewave(ts):
  return [0.1, np.sin(2*np.pi*5*(ts-1)/10000)] + np.random.normal(loc=[0.0,0.0], scale=0.1)

def normal_iid(ts):
  return np.random.normal(loc=[0.0, 0.0], scale=0.1)

def linear_classifier(x, theta):
  return np.sign(np.dot(x, theta))

class ContextualBanditFunctionalContext(Environment):
  """Simple linear contextual bandit with noisy contexts generated by given functions.
  E[y(t)|x(t)] = theta*x(t), x(t) = f(t) + eps.
  Action: (a(t),y^(t)), a(t): 0 or 1 for sample or not.
  contextfn: x(t), gives numpy array output"""

  def __init__(self, contextfn, true_theta, true_model, sample_cost=0.0):
    self.contextfn = contextfn
    self.true_theta = np.array(true_theta)
    self.true_model = true_model
    self.sample_cost = sample_cost

    self.step = 1 # time step for y(t), x(t)
    self.x_t = self.contextfn(self.step)
    self.optimal_reward = 1

    self.num_query = 0

  def get_observation(self):
    self.x_t = self.contextfn(self.step)
    return self.x_t

  def get_optimal_reward(self):
    return 1 # correct y prediction

  def get_expected_reward(self, action):
    query, pred = action
    exp_reward = -1 * self.sample_cost * int(query!=0) + int(self.true_model(self.x_t, self.true_theta) * pred > 0) # 1 - 0/1 loss
    return exp_reward

  def get_stochastic_reward(self, action):
    assert len(action)==2
    query, _ = action
    if query==0:
      reward = None # not sampled
    else:
      reward = self.true_model(self.x_t, self.true_theta) # true label

    return reward

  def advance(self, action, reward):
    """Updating the environment (useful for nonstationary bandit)."""
    if np.abs(np.dot(self.x_t, self.true_theta)) > 1:
      print("Error: Norm of theta exceeds 1. Reduce true theta", self.x_t, self.true_theta, np.dot(self.x_t, self.true_theta))

    self.step+=1

    self.num_query+=action[0]

class FiniteArmedBernoulliBandit(Environment):
  """Simple N-armed bandit."""

  def __init__(self, probs):
    self.probs = np.array(probs)
    assert np.all(self.probs >= 0)
    assert np.all(self.probs <= 1)

    self.optimal_reward = np.max(self.probs)
    self.n_arm = len(self.probs)

  def get_observation(self):
    return self.n_arm

  def get_optimal_reward(self):
    return self.optimal_reward

  def get_expected_reward(self, action):
    return self.probs[action]

  def get_stochastic_reward(self, action):
    return np.random.binomial(1, self.probs[action])

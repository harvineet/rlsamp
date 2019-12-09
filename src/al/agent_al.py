"""
Adapted from code by Ian Osband
https://github.com/iosband/ts_tutorial/

Selective sampling agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy.linalg import pinv

from base.agent import Agent
from base.agent import random_argmax

_SMALL_NUMBER = 1e-10
##############################################################################

class SelectiveSampleBBQ(Agent):
  """Agent to pick actions using BBQ (Bound on Bias Query)
  selective sampling method.
  http://cesa-bianchi.di.unimi.it/Pubblicazioni/icml2009.pdf"""
  
  def __init__(self, n_feat, kappa):
    self.kappa = kappa
    self.A_t = np.eye(n_feat)
    self.w_t = np.zeros(n_feat)
    self.step = 1
    self.logger = None
  
  def update_observation(self, observation, action, reward):
    self.step += 1

    x_t = observation
    y_t = reward
    if y_t is not None:
      A_tp1 = self.A_t + np.outer(x_t, x_t) # computed twice
      w_tp1 = np.dot(pinv(A_tp1), np.dot(self.A_t, self.w_t) + y_t * x_t)
      self.A_t = A_tp1
      self.w_t = w_tp1
    else:
      pass # same A_t, w_t

  def pick_action(self, observation):
    """Make prediction and decide to query."""
    x_t = observation
    pred = np.sign(np.dot(x_t, self.w_t))
    A_tp1 = self.A_t + np.outer(x_t, x_t) # A_{t+1}
    r_t = np.dot(x_t.T, np.dot(pinv(A_tp1), x_t)) # no effect of transpose
    query = int(r_t > self.step**(-self.kappa))

    action = (query, pred)

    self.logger = self.w_t

    return action

  def __str__(self):
    return "theta:{}, step:{}".format(self.w_t, self.step)


##############################################################################


class UniformRandom(Agent):
  """Simple agent to pick actions at random uniformly."""
  
  def __init__(self, n_arm):
    self.n_arm = n_arm
    self.logger = None
  
  def update_observation(self, observation, action, reward):
    # Naive error checking for compatibility with environment
    assert observation == self.n_arm
      
  def pick_action(self, observation):
    """Take random action prob epsilon, else be greedy."""
    action = np.random.randint(self.n_arm)

    return action


##############################################################################


class FiniteBernoulliBanditEpsilonGreedy(Agent):
  """Simple agent made for finite armed bandit problems."""

  def __init__(self, n_arm, a0=1, b0=1, epsilon=0.0):
    self.n_arm = n_arm
    self.epsilon = epsilon
    self.prior_success = np.array([a0 for arm in range(n_arm)])
    self.prior_failure = np.array([b0 for arm in range(n_arm)])
    self.logger = None

  def set_prior(self, prior_success, prior_failure):
    # Overwrite the default prior
    self.prior_success = np.array(prior_success)
    self.prior_failure = np.array(prior_failure)

  def get_posterior_mean(self):
    return self.prior_success / (self.prior_success + self.prior_failure)

  def get_posterior_sample(self):
    return np.random.beta(self.prior_success, self.prior_failure)

  def update_observation(self, observation, action, reward):
    # Naive error checking for compatibility with environment
    assert observation == self.n_arm

    if np.isclose(reward, 1):
      self.prior_success[action] += 1
    elif np.isclose(reward, 0):
      self.prior_failure[action] += 1
    else:
      raise ValueError('Rewards should be 0 or 1 in Bernoulli Bandit')

  def pick_action(self, observation):
    """Take random action prob epsilon, else be greedy."""
    if np.random.rand() < self.epsilon:
      action = np.random.randint(self.n_arm)
    else:
      posterior_means = self.get_posterior_mean()
      action = random_argmax(posterior_means)

    return action


##############################################################################


class FiniteBernoulliBanditTS(FiniteBernoulliBanditEpsilonGreedy):
  """Thompson sampling on finite armed bandit."""

  def pick_action(self, observation):
    """Thompson sampling with Beta posterior for action selection."""
    sampled_means = self.get_posterior_sample()
    action = random_argmax(sampled_means)
    return action
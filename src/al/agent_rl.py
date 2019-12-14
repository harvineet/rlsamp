"""
Adapted from code by Ian Osband
https://github.com/iosband/ts_tutorial/
Policy gradient methods adapted from code in PyTorch example
https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py

Finite bandit agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy.linalg import pinv
from collections import namedtuple

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from base.agent import Agent
from base.agent import random_argmax

_SMALL_NUMBER = 1e-7 #np.finfo(np.float32).eps.item()
##############################################################################

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class PolicyGradientActorCritic(Agent):
  """Agent to pick actions using policy gradient method."""
  
  def __init__(self, Policy, args):
    self.args = args
    # Policy
    self.policy = Policy(self.args.in_dim, self.args.n_act) # in and out dim of NN
    if self.args.optim == 'adam':
      self.optimizer = optim.Adam(self.policy.parameters(), lr=self.args.learn_rate)
    elif self.args.optim == 'adagrad':
      self.optimizer = optim.Adagrad(self.policy.parameters(), lr=self.args.learn_rate)
    else:
      self.optimizer = optim.SGD(self.policy.parameters(), lr=self.args.learn_rate, momentum=self.args.momentum)
    # Model
    self.A_t = np.eye(self.args.n_feat)
    self.w_t = np.zeros(self.args.n_feat)

    self.step = 1
    self.logger = None
  
  def finish_episode(self):
    """
    Training code. Calcultes actor and critic loss and performs backprop.
    """
    R = 0.
    saved_actions = self.policy.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in self.policy.rewards[::-1]:
        # calculate the discounted value
        R = r + self.args.gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    if len(returns) > 1:
      returns = (returns - returns.mean()) / (returns.std() + _SMALL_NUMBER)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss 
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    # reset gradients
    self.optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    self.optimizer.step()

    # reset rewards and action buffer
    del self.policy.rewards[:]
    del self.policy.saved_actions[:]

  def update_observation(self, observation, action, reward):
    self.step += 1

    x_t = observation
    y_t = reward
    _, pred = action
    if y_t is not None: # or query!=0
      self.policy.rewards.append(-1*self.args.sample_cost + int(y_t * pred > 0))

      A_tp1 = self.A_t + np.outer(x_t, x_t) # computed twice
      w_tp1 = np.dot(pinv(A_tp1), np.dot(self.A_t, self.w_t) + y_t * x_t)
      self.A_t = A_tp1
      self.w_t = w_tp1
    else:
      self.policy.rewards.append(0) # 0 reward
      # same A_t, w_t

    self.finish_episode() # change weights of policy

  def pick_action(self, observation):
    """Make prediction and decide to query."""
    x_t = observation
    pred = np.sign(np.dot(x_t, self.w_t))
    A_tp1 = self.A_t + np.outer(x_t, x_t) # A_{t+1}
    r_t = np.dot(x_t.T, np.dot(pinv(A_tp1), x_t)) # no effect of transpose
    # query = int(r_t > self.step**(-self.kappa))

    observation = np.append(x_t, r_t) # include variance from RLS
    x_t = torch.from_numpy(observation).float()
    probs, state_value = self.policy(x_t)
    m = Categorical(probs)
    a_t = m.sample() # 0 or 1
    self.policy.saved_actions.append(SavedAction(m.log_prob(a_t), state_value))
    query = a_t.item()

    # if self.step%100==0:
    #   print('pick_action', self.w_t[0:2])

    action = (query, pred)

    self.logger = self.w_t

    return action

  def __str__(self):
    return "theta:{}, step:{}".format(self.w_t, self.step)

class PolicyGradientREINFORCE(Agent):
  """Agent to pick actions using policy gradient method."""
  
  def __init__(self, Policy, args):
    self.args = args
    # Policy
    self.policy = Policy(self.args.in_dim, self.args.n_act) # in and out dim of NN
    if self.args.optim == 'adam':
      self.optimizer = optim.Adam(self.policy.parameters(), lr=self.args.learn_rate)
    else:
      self.optimizer = optim.SGD(self.policy.parameters(), lr=self.args.learn_rate, momentum=self.args.momentum)
    # Model
    self.A_t = np.eye(self.args.n_feat)
    self.w_t = np.zeros(self.args.n_feat)

    self.step = 1
    self.logger = None
  
  def finish_episode(self):
    R = 0
    policy_loss = []
    returns = []
    for r in self.policy.rewards[::-1]:
        R = r + self.args.gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns).float()
    if len(returns) > 1:
      returns = (returns - returns.mean()) / (returns.std() + _SMALL_NUMBER)
    for log_prob, R in zip(self.policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    self.optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    self.optimizer.step()
    # print('finish_episode', self.policy.rewards, policy_loss)
    del self.policy.rewards[:]
    del self.policy.saved_log_probs[:]

  def update_observation(self, observation, action, reward):
    self.step += 1

    x_t = observation
    y_t = reward
    _, pred = action
    if y_t is not None: # or query!=0
      self.policy.rewards.append(-1*self.args.sample_cost + int(y_t * pred > 0))

      A_tp1 = self.A_t + np.outer(x_t, x_t) # computed twice
      w_tp1 = np.dot(pinv(A_tp1), np.dot(self.A_t, self.w_t) + y_t * x_t)
      self.A_t = A_tp1
      self.w_t = w_tp1
    else:
      self.policy.rewards.append(0) # 0 reward
      # same A_t, w_t

    self.finish_episode() # change weights of policy

  def pick_action(self, observation):
    """Make prediction and decide to query."""
    x_t = observation
    pred = np.sign(np.dot(x_t, self.w_t))
    A_tp1 = self.A_t + np.outer(x_t, x_t) # A_{t+1}
    r_t = np.dot(x_t.T, np.dot(pinv(A_tp1), x_t)) # no effect of transpose
    # query = int(r_t > self.step**(-self.kappa))

    observation = np.append(x_t, r_t) # include variance from RLS
    x_t = torch.from_numpy(observation).float().unsqueeze(0)
    probs = self.policy(x_t)
    m = Categorical(probs)
    a_t = m.sample() # 0 or 1
    self.policy.saved_log_probs.append(m.log_prob(a_t))
    query = a_t.item()

    # if self.step%100==0:
    #   print('pick_action', self.w_t[0:2])

    action = (query, pred)

    self.logger = np.sum(self.w_t**2) # l2-norm squared

    return action

  def __str__(self):
    return "theta:{}, step:{}".format(self.w_t, self.step)

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
"""
Adapted from code in PyTorch example
https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py

Simple neural network policy."""

import torch.nn as nn
import torch.nn.functional as F

class PolicyNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PolicyNN, self).__init__()
        self.affine1 = nn.Linear(in_dim, 4)
        # self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(4, 2)
        # self.dropout2 = nn.Dropout(p=0.6)
        self.affine3 = nn.Linear(2, out_dim)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        # x = self.dropout(x)
        x = F.relu(x)
        x = self.affine2(x)
        # x = self.dropout2(x)
        x = F.relu(x)
        action_scores = self.affine3(x)
        return F.softmax(action_scores, dim=1)

class PolicyNNActorCritic(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self, in_dim, out_dim):
        super(PolicyNNActorCritic, self).__init__()
        self.affine1 = nn.Linear(in_dim, 2)
        # self.dropout1 = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(2, 2)

        # actor's layer
        self.action_head = nn.Linear(2, out_dim)

        # critic's layer
        self.value_head = nn.Linear(2, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = self.affine1(x)
        # x = self.dropout(x)
        x = F.relu(x)
        x = self.affine2(x)
        x = F.relu(x)

        # actor: choses action to take from state s_t 
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tupel of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t 
        return action_prob, state_values
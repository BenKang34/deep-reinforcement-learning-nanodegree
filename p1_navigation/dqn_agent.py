import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """

        pass

    def step(self):
        """Gather experience for each step and learn from it
        Params
        ======
        """
        pass

    def act(self):
        """Get action from on/off policy

        Params
        ======


        """
        pass

    def learn(self):
        """Learn from sample experiences and update weights for both target and local models

        Params
        ======


        """

    def update(self):
        pass



class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """

    def add(self):
        """Add a new experience to memory."""
        pass

    def sample(self):
        """Sample a batch of experiences from memory."""
        pass

    def __len__(self):
        """Return the current size of internal memory."""
        pass

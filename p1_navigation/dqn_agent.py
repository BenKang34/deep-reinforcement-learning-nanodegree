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
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.Q_model_local = QNetwork(state_size, action_size, seed).to(device)
        self.Q_model_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.Q_model_local.parameters(), lr=LR)

        # Replay Memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step
        self.t_step = 0


    def step(self, state, action, reward, next_state, done, double=False):
        """Gather experience for each step and learn from it
        Params
        ======
        """
        # Save experience to Replay Buffer
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) >= BATCH_SIZE:
                self.learn(self.memory.sample(), GAMMA, double)

    def act(self, state, epsilon=0.0):
        """Get action from on/off policy

        Params
        ======
            state (array_like): current state
            epsilon (float): for epsilon-greedy action selection
        """
        _state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        self.Q_model_local.eval()
        with torch.no_grad():
            action_values = self.Q_model_local(_state)
        self.Q_model_local.train()

        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma, double=False):
        """Learn from sample experiences and update weights for both target and local models

        Params
        ======
            experiences (*array_like): (s, a, r, s', done)
            gamma (float): discount rate
        """
        states, actions, rewards, next_states, dones = experiences
        self.Q_model_local.train()
        self.Q_model_target.eval()
        q_expected = self.Q_model_local(states).gather(1, actions)

        if double:
            self.Q_model_local.eval()
            with torch.no_grad():
                _, next_actions = self.Q_model_local(next_states).max(1)
                q_target_next = self.Q_model_target(next_states).gather(1, next_actions.unsqueeze(1).long())
            self.Q_model_local.train()
        else:
            q_target_next = self.Q_model_target(next_states).detach().max(1)[0].unsqueeze(1)
        q_target = rewards + gamma * q_target_next * (1-dones)

        loss = F.mse_loss(q_target, q_expected)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update(self.Q_model_local, self.Q_model_target, TAU)


    def update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """

        for local_params, target_params in zip(local_model.parameters(), target_model.parameters()):
            target_params.data.copy_(tau*local_params.data + (1.-tau)*target_params.data)



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
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        _experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(_experience)

    def sample(self):
        """Sample a batch of experiences from memory."""
        # Let's start with random sampling
        batch = random.sample(self.memory, k=self.batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for e in batch:
            states.append(e.state)
            actions.append(e.action)
            rewards.append(e.reward)
            next_states.append(e.next_state)
            dones.append(e.done)

        return (
            torch.from_numpy(np.vstack(states)).float().to(device),
            torch.from_numpy(np.vstack(actions)).long().to(device),
            torch.from_numpy(np.vstack(rewards)).float().to(device),
            torch.from_numpy(np.vstack(next_states)).float().to(device),
            torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(device)
        )

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

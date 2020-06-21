import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = 131072 #int(1e5)  # replay buffer size
BATCH_SIZE = 128 #64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, prioritized=False):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            prioritized (bool): whether to use proportional prioritized experience replay
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.prioritized = prioritized
        if self.prioritized:
            self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        else:
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

        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(_state)
        self.qnetwork_local.train()

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
        if self.prioritized:
            states, actions, rewards, next_states, dones, is_weights, sample_idx = experiences
        else:
            states, actions, rewards, next_states, dones = experiences
        self.qnetwork_local.train()
        self.qnetwork_target.eval()
        q_expected = self.qnetwork_local(states).gather(1, actions)

        if double:
            self.qnetwork_local.eval()
            with torch.no_grad():
                _, next_actions = self.qnetwork_local(next_states).max(1)
                q_target_next = self.qnetwork_target(next_states).gather(1, next_actions.unsqueeze(1).long())
            self.qnetwork_local.train()
        else:
            q_target_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        q_target = rewards + gamma * q_target_next * (1-dones)

        if self.prioritized:
            diff = q_target-q_expected
            loss = 0.5*torch.pow(diff, 2) # Mean Square Error
            loss = (is_weights*loss).mean()
            self.memory.update_priority(diff.abs().detach().squeeze(1).cpu().data.numpy(), sample_idx)
        else:
            loss = F.mse_loss(q_target, q_expected)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update(self.qnetwork_local, self.qnetwork_target, TAU)


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

    def save(self, path):
        """Save model parameters.

        Params
        ======
            path (str): path to save a model, torch model with extension of ".pt" or ".pth"
        """
        torch.save(self.qnetwork_local.state_dict(), path)
        print("Model saved as {}".format(path))

    def load(self, path, device='cpu'):
        """Load model parameters.

        Params
        ======
            path (str): path to load a model, torch model with extension of ".pt" or ".pth"
        """
        self.qnetwork_local.load_state_dict(torch.load(path, map_location=device))
        print("Model loaded from {} on {}".format(path, device))



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


class PrioritizedReplayBuffer(ReplayBuffer):
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, alpha=0.6, beta=0.4, epsilon=1e5):
        """Initialize a Prioritized ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        super(PrioritizedReplayBuffer, self).__init__(action_size, buffer_size, batch_size, seed)

        # Proportional Prioritization
        self.buffer_size = buffer_size
        self.error_abs = []
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        super(PrioritizedReplayBuffer, self).add(state, action, reward, next_state, done)
        self.error_abs.append(max(self.error_abs) if self.error_abs else 1)
        if len(self.error_abs) > self.buffer_size:
            self.error_abs.pop(0)


    def sample(self):
        """Sample a batch of experiences from memory by proportional prioritization."""

        # Stratified Sampling
        N = min(self.buffer_size, len(self.memory))
        n_segments, mod = divmod(N, self.batch_size)
        n_segments += 1 if mod else 0
        sample_idx = []
        priority_alpha = np.power(np.array(self.error_abs)+self.epsilon, self.alpha)
        priority_distribution = priority_alpha/np.sum(priority_alpha)
        importance_sampling_weights = np.power(N * priority_distribution, -self.beta)
        importance_sampling_weights /= importance_sampling_weights.max()
        for i in range(n_segments):
            min_idx, max_idx = i*self.batch_size, min((i+1)*self.batch_size, len(self.memory))
            sample_idx.append(random.choices(range(min_idx, max_idx), weights=priority_distribution[min_idx:max_idx], k=1)[0])
        experiences = [self.memory[i] for i in sample_idx]
        is_weights = importance_sampling_weights[sample_idx]

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        is_weights = torch.from_numpy(is_weights.reshape(-1,1)).float().to(device)

        return (states, actions, rewards, next_states, dones, is_weights, sample_idx)

    def update_priority(self, error_abs, sample_idx):
        for i, idx in enumerate(sample_idx):
            self.error_abs[idx] = error_abs[i]

        self.beta = min(self.beta + 5e5, 1)

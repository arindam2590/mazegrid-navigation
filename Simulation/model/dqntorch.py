import json
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from collections import deque


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque([], maxlen=capacity)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]


# ---------------------------------------------------------------------------
# Neural Network Modules  (names unchanged, now architecturally dynamic)
# ---------------------------------------------------------------------------
class DQN(nn.Module):
    """Fully-connected Q-network. Hidden layer widths are read from params."""
    def __init__(self, state_size, action_size, hidden_layers):
        super().__init__()
        sizes  = [state_size] + hidden_layers + [action_size]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:          # no activation on output layer
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DuelDQN(nn.Module):
    """Dueling Q-network. Shared trunk + value/advantage streams from params."""
    def __init__(self, state_size, action_size, shared_layers, value_layers, advantage_layers):
        super().__init__()

        # Shared trunk
        trunk_sizes = [state_size] + shared_layers
        trunk = []
        for i in range(len(trunk_sizes) - 1):
            trunk.append(nn.Linear(trunk_sizes[i], trunk_sizes[i + 1]))
            trunk.append(nn.ReLU())
        self.shared = nn.Sequential(*trunk)
        trunk_out = shared_layers[-1]

        # Value stream
        v_sizes = [trunk_out] + value_layers + [1]
        v_layers = []
        for i in range(len(v_sizes) - 1):
            v_layers.append(nn.Linear(v_sizes[i], v_sizes[i + 1]))
            if i < len(v_sizes) - 2:
                v_layers.append(nn.ReLU())
        self.value_stream = nn.Sequential(*v_layers)

        # Advantage stream
        a_sizes = [trunk_out] + advantage_layers + [action_size]
        a_layers = []
        for i in range(len(a_sizes) - 1):
            a_layers.append(nn.Linear(a_sizes[i], a_sizes[i + 1]))
            if i < len(a_sizes) - 2:
                a_layers.append(nn.ReLU())
        self.advantage_stream = nn.Sequential(*a_layers)

    def forward(self, x):
        shared    = self.shared(x)
        value     = self.value_stream(shared)
        advantage = self.advantage_stream(shared)
        return value + (advantage - advantage.mean(dim=-1, keepdim=True))


# ---------------------------------------------------------------------------
# Shared base — one place for __init__, remember, act, encode_state, etc.
# ---------------------------------------------------------------------------
class _BaseDQNModel:
    def __init__(self, state_size, action_size, maze, device):
        param_dir = 'Simulation/Utils/'
        with open(param_dir + 'train_params.json', 'r') as f:
            self.params = json.load(f)
        with open(param_dir + 'network_params.json', 'r') as f:
            net_params = json.load(f)
        with open(param_dir + 'sim_params.json', 'r') as f:
            sim_params = json.load(f)

        self.state_size  = state_size
        self.action_size = action_size
        self.device      = device
        self.maze        = maze

        self.replay_buffer = ReplayBuffer(self.params['BUFFER_SIZE'])
        self.gamma         = self.params['GAMMA']
        self.alpha         = self.params['ALPHA']
        self.epsilon       = self.params['EPSILON']
        self.epsilon_min   = self.params['EPSILON_MIN']
        self.epsilon_decay = self.params['EPSILON_DECAY']
        self.update_rate   = self.params['UPDATE_RATE']

        # Select architecture by size + difficulty (e.g. "60_complex")
        difficulty  = sim_params.get('DIFFICULTY', 'simple')
        arch_key    = f"{maze.maze_size}_{difficulty}"
        arch_dict   = net_params['NETWORK_ARCH']
        self.arch   = arch_dict.get(arch_key, arch_dict['default'])
        print(f'Info: Network arch key="{arch_key}"  hidden={self.arch.get("hidden", self.arch.get("shared"))}')

    # -----------------------------------------------------------------------
    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        with torch.no_grad():
            return self.main_network(
                self.encode_state(state).to(self.device)
            ).argmax().item()

    def encode_state(self, state):
        one_hot = torch.zeros(self.state_size, dtype=torch.float32, device=self.device)
        one_hot[state] = 1.0
        target_idx = (self.maze.destination[0] * self.maze.maze_size
                      + self.maze.destination[1])
        one_hot[target_idx] = 2.0
        return one_hot

    def _encode_batch(self, states):
        """Encode a list of integer states into a (N, state_size) tensor."""
        batch = torch.zeros(len(states), self.state_size,
                            dtype=torch.float32, device=self.device)
        for i, s in enumerate(states):
            batch[i, s] = 1.0
        target_idx = (self.maze.destination[0] * self.maze.maze_size
                      + self.maze.destination[1])
        batch[:, target_idx] = 2.0
        return batch

    def update_target_network(self):
        self.target_network.load_state_dict(self.main_network.state_dict())

    def _unpack_batch(self, minibatch):
        states     = [t[0] for t in minibatch]
        actions    = torch.tensor([t[1] for t in minibatch], device=self.device, dtype=torch.long)
        rewards    = torch.tensor([t[2] for t in minibatch], device=self.device, dtype=torch.float32)
        next_states= [t[3] for t in minibatch]
        dones      = torch.tensor([t[4] for t in minibatch], device=self.device, dtype=torch.bool)
        s_batch    = self._encode_batch(states)
        ns_batch   = self._encode_batch(next_states)
        return s_batch, actions, rewards, ns_batch, dones


# ---------------------------------------------------------------------------
# DQN
# ---------------------------------------------------------------------------
class DQNModel(_BaseDQNModel):
    def __init__(self, state_size, action_size, maze, device):
        super().__init__(state_size, action_size, maze, device)
        hidden = self.arch['hidden']
        self.main_network   = DQN(state_size, action_size, hidden).to(device)
        self.target_network = DQN(state_size, action_size, hidden).to(device)
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.loss_fn   = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.main_network.parameters(), lr=self.alpha)

    def train(self, batch_size):
        minibatch = self.replay_buffer.sample(batch_size)
        s, actions, rewards, ns, dones = self._unpack_batch(minibatch)

        # Current Q-values from main network
        current_Q = self.main_network(s)                           # (B, A)
        predicted = current_Q.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

        # Target: r  +  γ · max_a Q_target(s', a)   for non-terminal
        with torch.no_grad():
            next_Q   = self.target_network(ns).max(dim=1).values  # (B,)
            target   = rewards + self.gamma * next_Q * (~dones)

        loss = self.loss_fn(predicted, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


# ---------------------------------------------------------------------------
# Double DQN
# ---------------------------------------------------------------------------
class DoubleDQNModel(_BaseDQNModel):
    def __init__(self, state_size, action_size, maze, device):
        super().__init__(state_size, action_size, maze, device)
        hidden = self.arch['hidden']
        self.main_network   = DQN(state_size, action_size, hidden).to(device)
        self.target_network = DQN(state_size, action_size, hidden).to(device)
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.loss_fn   = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.main_network.parameters(), lr=self.alpha)

    def train(self, batch_size):
        minibatch = self.replay_buffer.sample(batch_size)
        s, actions, rewards, ns, dones = self._unpack_batch(minibatch)

        # Current Q-values
        current_Q = self.main_network(s)
        predicted = current_Q.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target: main selects next action, target evaluates it
        with torch.no_grad():
            next_actions = self.main_network(ns).argmax(dim=1, keepdim=True)  # (B,1)
            next_Q       = self.target_network(ns).gather(1, next_actions).squeeze(1)
            target       = rewards + self.gamma * next_Q * (~dones)

        loss = self.loss_fn(predicted, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


# ---------------------------------------------------------------------------
# Dueling DQN  (uses Double-DQN target — best practice for Dueling arch)
# ---------------------------------------------------------------------------
class DuelingDQNModel(_BaseDQNModel):
    def __init__(self, state_size, action_size, maze, device):
        super().__init__(state_size, action_size, maze, device)
        shared    = self.arch['shared']
        value     = self.arch['value']
        advantage = self.arch['advantage']
        self.main_network   = DuelDQN(state_size, action_size, shared, value, advantage).to(device)
        self.target_network = DuelDQN(state_size, action_size, shared, value, advantage).to(device)
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.loss_fn   = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.main_network.parameters(), lr=self.alpha)

    def train(self, batch_size):
        minibatch = self.replay_buffer.sample(batch_size)
        s, actions, rewards, ns, dones = self._unpack_batch(minibatch)

        # Current Q-values from the main (dueling) network
        current_Q = self.main_network(s)
        predicted = current_Q.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double-DQN target with Dueling architecture
        with torch.no_grad():
            next_actions = self.main_network(ns).argmax(dim=1, keepdim=True)
            next_Q       = self.target_network(ns).gather(1, next_actions).squeeze(1)
            target       = rewards + self.gamma * next_Q * (~dones)

        loss = self.loss_fn(predicted, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

import json

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Trajectory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []

    def append(self, state, action, reward, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)

    def compute_returns(self, device, gamma=0.99):
        #discounted_rewards = []
        discounted_rewards = np.zeros_like(np.array(self.rewards), dtype=float)
        reward_to_go = 0.0
        for t in reversed(range(len(self.rewards))):
            reward_to_go = reward_to_go * gamma + self.rewards[t]
            #discounted_rewards.insert(0, reward_to_go)
            discounted_rewards[t] = reward_to_go
        #print(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean())/discounted_rewards.std()
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32, device=device).unsqueeze(1)
        #print(f'Discounted: {discounted_rewards}')
        #discounted_rewards = (discounted_rewards - discounted_rewards.mean())/discounted_rewards.std()

        return discounted_rewards

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()


class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.layer1 = nn.Linear(state_size, 32)
        self.layer2 = nn.Linear(32, 16)
        self.layer3 = nn.Linear(16, action_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=-1)
        return x


class ValueNetwork(nn.Module):
    def __init__(self, state_size):
        super(ValueNetwork, self).__init__()
        self.layer1 = nn.Linear(state_size, 32)
        self.layer2 = nn.Linear(32, 16)
        self.layer3 = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class REINFORCE:
    def __init__(self, state_size, action_size, maze, device):
        param_dir = 'Simulation/Utils/'
        with open(param_dir + 'config.json', 'r') as file:
            self.params = json.load(file)

        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.maze = maze
        self.gamma = self.params['GAMMA']
        self.alpha = self.params['ALPHA']

        self.policy_network = PolicyNetwork(state_size, action_size).to(device=device)
        self.value_network = ValueNetwork(state_size).to(device=device)
        self.value_loss_fn = nn.MSELoss()
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.alpha)
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=self.alpha)

        self.trajectory = Trajectory()

    def remember(self, state, action, reward, log_prob):
        self.trajectory.append(state, action, reward, log_prob)

    def select_action(self, state):
        probs = self.policy_network(self.encode_state(state).to(self.device))
        action = torch.multinomial(probs, 1).item()
        return action, probs

    def train(self):
        discounted_rewards_tensor = self.trajectory.compute_returns(self.device, self.gamma)
        state_value = []
        for state in self.trajectory.states:
            state_value.append(self.value_network(self.encode_state(state).to(self.device)))
        print(state_value)
        state_value_tensor = torch.stack(state_value).float()
        print(f'State Tensor:\n {state_value_tensor}')
        print(f'Reward Tensor:\n {discounted_rewards_tensor}')
        value_loss = self.value_loss_fn(state_value_tensor, discounted_rewards_tensor)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        advantages = discounted_rewards_tensor - state_value_tensor
        advantages = advantages.detach()
        log_probs_tensor = torch.stack(self.trajectory.log_probs)
        print(f"log_probs_tensor shape: \n{log_probs_tensor.shape},\n {log_probs_tensor}")
        print(f"advantages shape: \n{advantages.shape},\n {advantages}")

        policy_loss = -torch.sum(log_probs_tensor.to(self.device) * advantages.to(self.device))
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        return policy_loss, value_loss

    def encode_state(self, state):
        one_hot_tensor = torch.zeros(self.state_size, dtype=torch.float32, device=self.device)
        one_hot_tensor[state] = 1
        target_index = self.maze.destination[0] * self.maze.maze_size + self.maze.destination[1]
        one_hot_tensor[target_index] = 2
        return one_hot_tensor

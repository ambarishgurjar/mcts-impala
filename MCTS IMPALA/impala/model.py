import torch as T
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, n_observations, n_actions, device = None):
        super(Network, self).__init__()
        self.linear1 = nn.Linear(n_observations, 24)
        self.linear2 = nn.Linear(24, 24)
        self.linear3 = nn.Linear(24, 24)

        self.head = Head(24, n_actions)
        # self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu") if device is None else device
        self.to(self.device)

    def forward(self, states, actor=False):
        batch_size, trajectory_length, states = self.batch(states, actor)
        states = T.Tensor(states).to(self.device)
        states = F.relu(self.linear1(states))
        states = F.relu(self.linear2(states))
        states = states.view(batch_size, trajectory_length, -1)
        actions, values = self.head(states)

        # Why do we return the distribution over actions for the actor?
        if actor:
            return T.multinomial(F.softmax(actions.squeeze(0), dim = 1), num_samples=1).item(), actions.view(1, -1)
        return actions.view(batch_size, -1, trajectory_length), values.view(batch_size, trajectory_length)

    def batch(self, states, actor=False):
        """
        If called by learner, then create a column vector of size batch_size * trajectory_len * (state/action/reward)
        """
        if actor:
            return 1, 1, states  # , actions, rewards
        batch_size = states.shape[0]
        trajectory_length = states.shape[1]
        # https://stackoverflow.com/questions/46826218/pytorch-how-to-get-the-shape-of-a-tensor-as-a-list-of-int
        states = states.reshape(batch_size * trajectory_length, *states.shape[2:])
        # actions = actions.reshape(batch_size * trajectory_length, -1)
        # rewards = rewards.reshape(batch_size * trajectory_length, -1)
        return batch_size, trajectory_length, states  # , actions, rewards


class Head(nn.Module):
    def __init__(self, input_dim, action_space):
        super().__init__()
        self.actor_linear = nn.Linear(input_dim, action_space)
        self.critic_linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        actions = self.actor_linear(x)
        values = self.critic_linear(x)
        return actions, values

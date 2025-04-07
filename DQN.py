### Sourced from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#replay-memory
from torch import nn
import torch
from torch.nn import functional as F

class DQN(nn.Module):

    def __init__(self, n_states, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_states, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

        self.action_mapping = {}

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
    def get_action(self, x):
        return torch.argmax(self.forward(x), dim=1)

    def set_action_mapping(self, action_mapping):
        self.action_mapping = action_mapping

    def get_action_mapping(self):
        return self.action_mapping

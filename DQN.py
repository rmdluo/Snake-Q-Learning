### Sourced from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#replay-memory
from torch import nn
import torch
from torch.nn import functional as F

class DQN(nn.Module):

    def __init__(self, n_channels, board_height, board_width, n_actions):
        super(DQN, self).__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(n_channels, 16, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, 4, board_height, board_width)
            conv_out = self.backbone(dummy_input)
            self.flattened_size = conv_out.view(1, -1).size(1)

        self.layer1 = nn.Linear(self.flattened_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

        self.action_mapping = {}

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = self.backbone(x)
        x = F.relu(self.layer1(x.flatten(1)))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
    def get_action(self, x):
        return torch.argmax(self.forward(x), dim=1)

    def set_action_mapping(self, action_mapping):
        self.action_mapping = action_mapping

    def get_action_mapping(self):
        return self.action_mapping

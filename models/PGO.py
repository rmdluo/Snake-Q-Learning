### Partially sourced from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#replay-memory
from torch import nn
import torch
from torch.nn import functional as F
import numpy as np

from models.base_model import Model

class PGOModel(Model):
    def __init__(self, policy_network, device='cpu'):
        self.policy_network = policy_network
        self.device = device

    def parameters(self):
        return self.policy_network.parameters()
    
    def get_action_mapping(self):
        return self.policy_network.get_action_mapping()
    
    def get_action(self, state, action_check_fn=None, last_action=None):
        if action_check_fn:
            print("Warning: action_check_fn not used in PGO")

        # print(state.shape)

        probs, actions = self.policy_network.get_action(state)
        return probs, [self.get_action_mapping()[int(action)] for action in actions]
    
    # define one optimization step, largely inspired by https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#training-loop
    def optimize(self, samples, optimizer, criterion, config):
        # Convert to tensors
        log_probs, rewards = samples

        # compute the loss
        loss = criterion(log_probs, rewards)

        # step the weights
        optimizer.zero_grad();
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_network.parameters(), 100)
        optimizer.step()

        return loss.item()

    def save(self, model_save_path, optimizer, lr_schedule, config, step_count):
        torch.save({
            'policy_network_state_dict': self.policy_network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_schedule.state_dict(),
            'config': config,
        }, model_save_path)

class PGO(nn.Module):

    def __init__(self, n_channels, board_height, board_width, n_actions):
        super(PGO, self).__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(n_channels, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, n_channels, board_height, board_width)
            conv_out = self.backbone(dummy_input)
            self.flattened_size = conv_out.view(1, -1).size(1)

        self.layer1 = nn.Linear(self.flattened_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, n_actions)

        self.softmax = nn.Softmax()

        self.action_mapping = {}

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = self.backbone(x)
        x = F.relu(self.layer1(x.flatten(1)))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x, self.softmax(x)
    
    def get_action(self, x):
        _, probs = self.forward(x)
        samples = torch.multinomial(probs, num_samples=1).squeeze(1)
        return probs.gather(1, samples.unsqueeze(1)).squeeze(1), samples
    
    def get_action_rankings(self, x):
        return torch.argsort(self.forward(x), dim=1, descending=True)

    def set_action_mapping(self, action_mapping):
        self.action_mapping = action_mapping

    def get_action_mapping(self):
        return self.action_mapping
    
    @staticmethod
    def get_model(config, state_size_width, state_size_height, number_of_actions, actions, device='cpu'):
        # Initialize models
        policy_network = PGO(config['state_history_size'], state_size_width, state_size_height, number_of_actions).to(device)

        # Set up each model's action mapping
        action_mapping = {i: action for i, action in enumerate(actions)}
        for i, action in enumerate(actions):
            action_mapping[action] = i
        policy_network.set_action_mapping(action_mapping)

        return PGOModel(policy_network, device=device)
    
    @staticmethod
    def get_optimizer(model: PGOModel, config):
        return torch.optim.AdamW(model.parameters(), lr=config['lr'], amsgrad=True)

    @staticmethod
    def get_criterion(gamma):
        class PGOLoss(nn.Module):
            def __init__(self, gamma):
                super(PGOLoss, self).__init__()
                self.gamma = gamma

            def forward(self, log_probs, rewards):
                _, T = rewards.shape
                discounted_rewards = torch.zeros_like(rewards)

                # Compute discounted cumulative returns G_t
                for t in reversed(range(T)):
                    if t == T - 1:
                        discounted_rewards[:, t] = rewards[:, t]
                    else:
                        discounted_rewards[:, t] = rewards[:, t] + self.gamma * discounted_rewards[:, t + 1]
                loss = -(log_probs * discounted_rewards).mean()
                return loss
        return PGOLoss(gamma)

    @staticmethod
    def get_lr_scheduler(optimizer, config):
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['epochs'], 1e-5)

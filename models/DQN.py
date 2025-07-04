### Partially sourced from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#replay-memory
from torch import nn
import torch
from torch.nn import functional as F
import numpy as np

from models.base_model import Model

class DQNModel(Model):
    def __init__(self, policy_network, target_network, device='cpu'):
        self.policy_network = policy_network
        self.target_network = target_network
        self.device = device

    def parameters(self):
        return self.policy_network.parameters()
    
    def get_action_mapping(self):
        return self.policy_network.get_action_mapping()
    
    def get_action(self, state, action_check_fn=None, last_action=None):
        if action_check_fn and not last_action:
            raise Exception("Passed in action check function without last action")

        with torch.no_grad():
            action_rankings = self.policy_network.get_action_rankings(state)
            action_rankings = action_rankings.squeeze()
            # action = policy_network.get_action_mapping()[int(policy_network.get_action(torch.tensor([state_history], device=device, dtype=torch.float32)))]
            for j in range(len(action_rankings)):
                action = self.policy_network.get_action_mapping()[int(action_rankings[j])]
                if not action_check_fn or action_check_fn(action, last_action):
                    break
        return action
    
    # define one optimization step, largely inspired by https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#training-loop
    def optimize(self, samples, optimizer, criterion, config):
        # Convert to tensors
        state_batch = torch.tensor(np.array(samples.state), device=self.device, dtype=torch.float32)
        action_batch = torch.tensor(samples.action, device=self.device)
        reward_batch = torch.tensor(samples.reward, device=self.device, dtype=torch.float32)
        next_state_batch = torch.tensor(np.array(samples.next_state), device=self.device, dtype=torch.float32)

        # process using the policy network
        non_final_mask = torch.ones(config['batch_size'], device=self.device, dtype=torch.bool)
        non_final_next_states = next_state_batch
        state_action_values = self.policy_network(state_batch).gather(1, action_batch.unsqueeze(1))

        next_state_values = torch.zeros(config['batch_size'], device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1).values
        expected_state_action_values = (next_state_values * config['action_discount_factor']) + reward_batch

        # compute the loss
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # step the weights
        optimizer.zero_grad();
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_network.parameters(), 100)
        optimizer.step()

        # make a soft update from the policy network to the target network
        target_network_state_dict = self.target_network.state_dict()
        policy_network_state_dict = self.policy_network.state_dict()
        for key in policy_network_state_dict:
            target_network_state_dict[key] = policy_network_state_dict[key]*config['soft_update_tau'] + target_network_state_dict[key]*(1-config['soft_update_tau'])
        self.target_network.load_state_dict(target_network_state_dict)

    def save(self, model_save_path, optimizer, lr_schedule, config, step_count):
        torch.save({
            'policy_network_state_dict': self.policy_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_schedule.state_dict(),
            'config': config,
            'step_count': step_count
        }, model_save_path)

class DQN(nn.Module):

    def __init__(self, n_channels, board_height, board_width, n_actions):
        super(DQN, self).__init__()

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
    
    def get_action_rankings(self, x):
        return torch.argsort(self.forward(x), dim=1, descending=True)

    def set_action_mapping(self, action_mapping):
        self.action_mapping = action_mapping

    def get_action_mapping(self):
        return self.action_mapping
    
    @staticmethod
    def get_model(config, state_size_width, state_size_height, number_of_actions, actions, device='cpu'):
        # Initialize models
        policy_network = DQN(config['state_history_size'], state_size_width, state_size_height, number_of_actions).to(device)
        target_network = DQN(config['state_history_size'], state_size_width, state_size_height, number_of_actions).to(device)
        target_network.load_state_dict(policy_network.state_dict()) # do this to make sure nets are exact replicas

        # Set up each model's action mapping
        action_mapping = {i: action for i, action in enumerate(actions)}
        for i, action in enumerate(actions):
            action_mapping[action] = i
        policy_network.set_action_mapping(action_mapping)
        target_network.set_action_mapping(action_mapping)

        return DQNModel(policy_network, target_network, device=device)
    
    @staticmethod
    def get_optimizer(model: DQNModel, config):
        return torch.optim.AdamW(model.parameters(), lr=config['lr'], amsgrad=True)

    @staticmethod
    def get_criterion():
        return torch.nn.SmoothL1Loss() # L2 above 1, L1 below 1

    @staticmethod
    def get_lr_scheduler(optimizer, config):
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['episodes'], 1e-5)

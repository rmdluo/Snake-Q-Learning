from collections import deque
import math
import os
import random
import shutil
import torch
import numpy as np
from replay_memory import ReplayMemory, Transition
from snake_env import SnakeEnv
from DQN import DQN

# Config
config = {
    'episodes': 10000,
    'max_steps': 2000,
    'lr': 1e-4,
    'memory_capacity': 1000000,
    'memory_prefill_percent': 0.10,
    'action_epsilon_start': 0.95,
    'action_epsilon_end': 0.05,
    'action_epsilon_decay': 1000,
    'soft_update_tau': 0.005,
    'action_discount_factor': 0.99,
    'state_history_size': 1,
    'batch_size': 64,
    'logging_folder': 'logs'
}
if os.path.exists(config['logging_folder']):
    shutil.rmtree(config['logging_folder'])
os.makedirs(config['logging_folder'])
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize the environment
env = SnakeEnv()
initial_state = env.reset()
state_size_height = len(initial_state)
state_size_width = len(initial_state[0])
number_of_actions = len(env.get_actions())

# Initialize models
policy_network = DQN(config['state_history_size'], state_size_width, state_size_height, number_of_actions).to(device)
target_network = DQN(config['state_history_size'], state_size_width, state_size_height, number_of_actions).to(device)
target_network.load_state_dict(policy_network.state_dict()) # do this to make sure nets are exact replicas

# Set up each model's action mapping
action_mapping = {i: action for i, action in enumerate(env.get_actions())}
for i, action in enumerate(env.get_actions()):
    action_mapping[action] = i
policy_network.set_action_mapping(action_mapping)
target_network.set_action_mapping(action_mapping)

# Initialize optimizer and loss function
optimizer = torch.optim.AdamW(policy_network.parameters(), lr=config['lr'], amsgrad=True)
criterion = torch.nn.SmoothL1Loss() # L2 above 1, L1 below 1
lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['episodes'], 1e-5)

# Initialize replay memory object
replay_memory = ReplayMemory(config['memory_capacity'])

# Prefill the replay memory
state = initial_state
last_action = env.get_dummy_action()
state_history = deque([state for _ in range(config['state_history_size'])], config['state_history_size'])
while replay_memory.get_percent_full() < config['memory_prefill_percent']:
    # sample a random action
    action = env.sample_action_no_die_no_opposite(last_action)

    # get the results of using the action
    next_state, reward, terminated = env.step(action)

    # move the next_state into state
    state = next_state
    current_state_history = list(state_history)
    state_history.append(state)
    next_state_history = list(state_history)

    # store the result in the replay memory
    replay_memory.push(
        np.array(current_state_history, dtype=np.uint8),
        policy_network.get_action_mapping()[action],
        reward,
        np.array(next_state_history, dtype=np.uint8)
    )

    # reset the environment if the state is terminal
    if terminated:
        state = env.reset()

# define one optimization step, largely inspired by https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#training-loop
def optimize(policy_network, target_network, optimizer, criterion, replay_memory: ReplayMemory, batch_size):
    # get a minibatch of transitions from the replay memory
    transitions = replay_memory.sample(batch_size)

    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Convert to tensors
    state_batch = torch.tensor(np.array(batch.state), device=device, dtype=torch.float32)
    action_batch = torch.tensor(batch.action, device=device)
    reward_batch = torch.tensor(batch.reward, device=device, dtype=torch.float32)
    next_state_batch = torch.tensor(np.array(batch.next_state), device=device, dtype=torch.float32)

    # process using the policy network
    non_final_mask = torch.ones(batch_size, device=device, dtype=torch.bool)
    non_final_next_states = next_state_batch
    state_action_values = policy_network(state_batch).gather(1, action_batch.unsqueeze(1))

    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_network(non_final_next_states).max(1).values
    expected_state_action_values = (next_state_values * config['action_discount_factor']) + reward_batch

    # compute the loss
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # step the weights
    optimizer.zero_grad();
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_network.parameters(), 100)
    optimizer.step()


# Train
step_count = 0
for i in range(config['episodes']):
    # reset the environment
    state = env.reset()
    last_action = env.get_dummy_action()
    culmulated_reward = 0
    state_history = deque([state for _ in range(config['state_history_size'])], config['state_history_size'])
    action_epsilon = config['action_epsilon_end'] + (config['action_epsilon_start'] - config['action_epsilon_end']) * math.exp(-1. * i / config['action_epsilon_decay'])

    # open logging file
    f = open(os.path.join(config['logging_folder'], f'{i}.txt'), mode='w')
    f.write(str(env) + "\n")

    # step until the state is terminal or reaches the max steps
    for _ in range(config['max_steps']):
        # choose an epsilon action
        random_selection = False
        if random.random() < action_epsilon:
            random_selection = True
            action = env.sample_action_no_die_no_opposite(last_action)
        else:
            with torch.no_grad():
                action_rankings = policy_network.get_action_rankings(torch.tensor([state_history], device=device, dtype=torch.float32))
                action_rankings = action_rankings.squeeze()
                # action = policy_network.get_action_mapping()[int(policy_network.get_action(torch.tensor([state_history], device=device, dtype=torch.float32)))]
                for j in range(len(action_rankings)):
                    action = policy_network.get_action_mapping()[int(action_rankings[j])]
                    if env.check_action(action, last_action):
                        break
        last_action = action

        # step the environment with that action
        next_state, reward, terminated = env.step(action)
        culmulated_reward += reward

        # update state
        state = next_state
        current_state_history = list(state_history)
        state_history.append(state)
        next_state_history = list(state_history)

        # store the result in the replay memory
        replay_memory.push(
            np.array(current_state_history, dtype=np.uint8),
            policy_network.get_action_mapping()[action],
            reward,
            np.array(next_state_history, dtype=np.uint8)
        )

        # call one optimize step
        optimize(policy_network, target_network, optimizer, criterion, replay_memory, config['batch_size'])

        # make a soft update from the policy network to the target network
        target_network_state_dict = target_network.state_dict()
        policy_network_state_dict = policy_network.state_dict()
        for key in policy_network_state_dict:
            target_network_state_dict[key] = policy_network_state_dict[key]*config['soft_update_tau'] + target_network_state_dict[key]*(1-config['soft_update_tau'])
        target_network.load_state_dict(target_network_state_dict)

        # increment the step count
        step_count += 1

        # add to logs
        f.write("-------------------------------\n")
        if random_selection:
            f.write("RANDOM ")
        f.write(action.name + "\n")
        f.write(str(env) + "\n")

        # break if the resulting state was terminal
        if terminated:
            f.close()
            print(i, action_epsilon, step_count, culmulated_reward, lr_schedule.get_last_lr())
            lr_schedule.step()
            break

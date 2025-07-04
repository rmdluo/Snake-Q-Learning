from collections import deque
import math
import os
import random
import shutil
import torch
import numpy as np
from replay_memory import ReplayMemory, Transition
from snake_env import SnakeEnv
from models.DQN import DQN

# Command line arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("config_file")
args = parser.parse_args()

# Config
import yaml
with open(args.config_file, "r") as file:
    config = yaml.safe_load(file)

# Logging setup
if os.path.exists(os.path.join(config['logging_folder'], config['test_name'])):
    shutil.rmtree(os.path.join(config['logging_folder'], config['test_name']))
run_save_path = os.path.join(config['logging_folder'], config['test_name'], 'runs')
weights_save_path = os.path.join(config['logging_folder'], config['test_name'], "weights")
os.makedirs(run_save_path, exist_ok=True)
os.makedirs(weights_save_path, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# print configs
from pprint import pprint

print("-" * 60)
print(f"Using the following configuration from {args.config_file}:")
pprint(config)
print("-" * 60)
print(f"Saving run logs to: {run_save_path}")
print(f"Saving model weights to: {weights_save_path}")
print("-" * 60)
print(f"Using device: {device}")
print("-" * 60)

# Initialize the environment
env = SnakeEnv()
initial_state = env.reset()
state_size_height = len(initial_state)
state_size_width = len(initial_state[0])
number_of_actions = len(env.get_actions())

if config['type'] == 'dqn':
    model = DQN.get_model(
        config,
        state_size_width,
        state_size_height,
        number_of_actions,
        env.get_actions(),
        device=device
    )

    # Initialize optimizer and loss function
    optimizer = DQN.get_optimizer(model, config)
    criterion = DQN.get_criterion()
    lr_schedule = DQN.get_lr_scheduler(optimizer, config)

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
        model.get_action_mapping()[action],
        reward,
        np.array(next_state_history, dtype=np.uint8)
    )

    # reset the environment if the state is terminal
    if terminated:
        state = env.reset()

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
    f = open(os.path.join(run_save_path, f'{i}.txt'), mode='w')
    f.write(str(env) + "\n")

    # step until the state is terminal or reaches the max steps
    for _ in range(config['max_steps']):
        # choose an epsilon action
        random_selection = False
        if random.random() < action_epsilon:
            random_selection = True
            action = env.sample_action_no_die_no_opposite(last_action)
        else:
            action = model.get_action(
                torch.tensor([state_history], device=model.device, dtype=torch.float32),
                action_check_fn=env.check_action,
                last_action=last_action
            )
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
            model.get_action_mapping()[action],
            reward,
            np.array(next_state_history, dtype=np.uint8)
        )

        # get a minibatch of transitions from the replay memory
        transitions = replay_memory.sample(config['batch_size'])

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # call one optimize step
        model.optimize(batch, optimizer, criterion, config)

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

# Save the trained policy and target network models, optimizer, and scheduler
model_save_path = os.path.join(weights_save_path, "final_checkpoint.pth")
model.save(model_save_path, optimizer, lr_schedule, config, step_count)
print(f"Model checkpoint saved to {model_save_path}")

from collections import deque
import math
import os
import random
import shutil
import torch
import numpy as np
import mlflow
from replay_memory import ReplayMemory, Transition
from snake_env import SnakeEnv
from models.PGO import PGO
from models.DQN import DQN

# Load .env file
from dotenv import load_dotenv
load_dotenv()

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
if os.path.exists(os.path.join(config['logging_folder'], config['experiment_name'])):
    shutil.rmtree(os.path.join(config['logging_folder'], config['experiment_name']))
run_save_path = os.path.join(config['logging_folder'], config['experiment_name'], 'runs')
weights_save_path = os.path.join(config['logging_folder'], config['experiment_name'], "weights")
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

per_step = False
if config['type'] == 'dqn':
    per_step = True

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
elif config['type'] == 'pgo':
    per_step = False
    model = PGO.get_model(
        config,
        state_size_width,
        state_size_height,
        number_of_actions,
        env.get_actions(),
        device=device
    )

    # Initialize optimizer and loss function
    optimizer = PGO.get_optimizer(model, config)
    criterion = PGO.get_criterion(config['action_discount_factor'])
    lr_schedule = PGO.get_lr_scheduler(optimizer, config)

# Initialize MLFlow
mlflow.set_tracking_uri(uri=os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(config["experiment_name"])

with mlflow.start_run():
    mlflow.log_params(config)

    if per_step:
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
                for j in range(config['max_steps']):
                    # choose an epsilon action
                    random_selection = False
                    if random.random() < action_epsilon:
                        random_selection = True
                        action = env.sample_action_no_die_no_opposite(last_action)
                    else:
                        action = model.get_action(
                            torch.tensor([state_history], device=model.device, dtype=torch.float32),
                            # action_check_fn=env.check_action,
                            # last_action=last_action
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
                    loss = model.optimize(batch, optimizer, criterion, config)

                    # increment the step count
                    step_count += 1

                    # add to logs
                    f.write("-------------------------------\n")
                    if random_selection:
                        f.write("RANDOM ")
                    f.write(action.name + "\n")
                    f.write(str(env) + "\n")

                    # break if the resulting state was terminal
                    if terminated or j == config['max_steps'] - 1:
                        f.close()
                        print(i, action_epsilon, step_count, culmulated_reward, lr_schedule.get_last_lr())
                        lr_schedule.step()
                        mlflow.log_metric("loss", loss, step=i)
                        mlflow.log_metric("score", culmulated_reward, step=i)
                        break
            f.close()
        
    else:
        envs = [SnakeEnv() for _ in range(config['batch_size'])]

        for epoch in range(config['epochs']):
            log_probs = torch.zeros((config['batch_size'], config['max_steps']), device=device, dtype=torch.float32)
            rewards = torch.zeros((config['batch_size'], config['max_steps']), device=device, dtype=torch.float32)

            states = [env.reset() for env in envs]
            logging_files = [open(os.path.join(run_save_path, f"epoch_{epoch}_run_{i}.txt"), "w") for i in range(config['batch_size'])]
            for i, f in enumerate(logging_files):
                f.write(str(envs[i]) + "\n")
            is_terminated = torch.tensor([False for _ in envs], device=device)
            state_histories = [deque([state for _ in range(config['state_history_size'])], config['state_history_size']) for state in states]

            for j in range(config['max_steps']):

                batch_states = torch.stack([
                    torch.stack([
                        torch.tensor(state, device=model.device, dtype=torch.float32) for state in history
                    ], dim=0)
                    for history in state_histories
                ])

                action_probs, actions = model.get_action(batch_states)

                # Store log probs for the episode
                log_probs[~is_terminated, j] = torch.log(action_probs)[~is_terminated]

                current_rewards = torch.empty(len(envs))
                for k in range(len(envs)):
                    if is_terminated[k]:
                        current_rewards[k] = 0
                        continue

                    state, reward, terminated = envs[k].step(actions[k])
                    is_terminated[k] = terminated
                    current_rewards[k] = reward
                    state_histories[k].append(state)

                    logging_files[k].write("-------------------------------\n")
                    logging_files[k].write(actions[k].name + "\n")
                    logging_files[k].write(str(envs[k]) + "\n")

                rewards[:, j] = current_rewards
                if is_terminated.all():
                    break

            for f in logging_files:
                f.close()

            loss = model.optimize((log_probs, rewards), optimizer, criterion, config)

            avg_score = rewards.sum(1).mean().item()
            print(epoch, avg_score, lr_schedule.get_last_lr())
            lr_schedule.step()
            mlflow.log_metric("loss", loss, step=epoch)
            mlflow.log_metric("avg_score", avg_score, step=epoch)

    # Save the trained policy and target network models, optimizer, and scheduler
    model_save_path = os.path.join(weights_save_path, "final_checkpoint.pth")
    model.save(model_save_path, optimizer, lr_schedule, config, step_count)
    print(f"Model checkpoint saved to {model_save_path}")

    # log to MLFlow
    mlflow.log_artifacts(run_save_path, artifact_path="runs")
    mlflow.log_artifacts(weights_save_path, artifact_path="weights")

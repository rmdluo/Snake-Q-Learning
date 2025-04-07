import math
import random
import torch
from replay_memory import ReplayMemory, Transition
from snake_env import SnakeEnv
from DQN import DQN

# Config
config = {
    'episodes': 5000,
    'max_steps': 2000,
    'lr': 1e-4,
    'memory_capacity': 100000,
    'memory_prefill_percent': 0.5,
    'action_epsilon_start': 0.95,
    'action_epsilon_end': 0.00,
    'action_epsilon_decay': 100,
    'soft_update_tau': 0.005,
    'action_discount_factor': 0.99,
    'batch_size': 1024
}
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize the environment
env = SnakeEnv()
initial_state = env.reset()
state_size = len(initial_state)
number_of_actions = len(env.get_actions())

# Initialize models
policy_network = DQN(state_size, number_of_actions).to(device)
target_network = DQN(state_size, number_of_actions).to(device)
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

# Initialize replay memory object
replay_memory = ReplayMemory(config['memory_capacity'])

# Prefill the replay memory
state = initial_state
last_action = env.get_dummy_action()
while replay_memory.get_percent_full() < config['memory_prefill_percent']:
    # sample a random action
    action = env.sample_action_with_checks(last_action)

    # get the results of using the action
    next_state, reward, terminated = env.step(action)

    # store it in the memory buffer
    replay_memory.push(
        torch.tensor([state], device=device, dtype=torch.float32),
        torch.tensor([policy_network.get_action_mapping()[action]], device=device),
        torch.tensor([reward], device=device, dtype=torch.float32),
        torch.tensor([next_state], device=device, dtype=torch.float32)
    )

    # move the next_state into state
    state = next_state

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

    # process using the policy network
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

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

    action_epsilon = config['action_epsilon_end'] + (config['action_epsilon_start'] - config['action_epsilon_end']) * math.exp(-1. * i / config['action_epsilon_decay'])

    # step until the state is terminal or reaches the max steps
    for _ in range(config['max_steps']):
        # choose an epsilon action
        if random.random() < action_epsilon:
            action = env.sample_action_with_checks(last_action)
        else:
            with torch.no_grad():
                action = policy_network.get_action_mapping()[int(policy_network.get_action(torch.tensor([state], device=device, dtype=torch.float32)))]

        # step the environment with that action
        next_state, reward, terminated = env.step(action)
        culmulated_reward += reward

        # store the result in the replay memory
        replay_memory.push(
            torch.tensor([state], device=device, dtype=torch.float32),
            torch.tensor([policy_network.get_action_mapping()[action]], device=device),
            torch.tensor([reward], device=device, dtype=torch.float32),
            torch.tensor([next_state], device=device, dtype=torch.float32)
        )

        # update state
        state = next_state

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

        # break if the resulting state was terminal
        if terminated:
            print(action_epsilon, step_count, culmulated_reward)
            break

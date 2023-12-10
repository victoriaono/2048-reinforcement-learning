import torch
import torch.nn as nn
import torch.optim as optim
import random
import sys
import numpy as np
from itertools import count

from game import Game2048
from dqn import DQN
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Neural Network initialization and utilities
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
NUM_EPISODES = 50000
BATCH_SIZE = 64
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
TARGET_UPDATE = 20
learning_rate = 1e-4
n_actions = 4

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
# policy_net.load_state_dict(torch.load('policy_net.pth'))
# target_net.load_state_dict(torch.load('target_net.pth'))
# target_net.eval()
# policy_net.train()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
memory = ReplayMemory(10000) 

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = max(EPS_END, EPS_START * (EPS_DECAY ** steps_done))
    # eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # max(1)[1] returns the index of the maximum output
            # get the max action value reshaped to [1, 1]
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    
losses = []

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    # In-place gradient clipping
    # torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)

    optimizer.step()
    losses.append(loss)

def train():
    game = Game2048()
    total_scores, best_tile_list = [], []

    # num_episodes = NUM_EPISODES
    print("total number of episodes:", NUM_EPISODES)
    for i_episode in range(NUM_EPISODES):
        print(f"Episode {i_episode}")
        game.reset()
        state = encode_state(game.matrix).to(torch.float32)
        
        for t in count():
            # Select and perform an action
            action = select_action(state)
            # old_score = game.get_sum()
            game.make_move(action.item())

            done = game.is_game_over()

            reward = game.get_merge_score()
            # reward = game.get_sum() - old_score
            reward = torch.tensor([reward], device=device)

            # Observe new state
            if not done:
                next_state = encode_state(game.matrix).to(torch.float32)
            else:
                next_state = None

            # if next_state != None and torch.eq(state, next_state).all():
            #     reward -= 10

            # Store the transition in memory
            if next_state == None or len(memory) == 0 or not same_move(state, next_state, memory.memory[-1]):
                memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            # optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            # target_net_state_dict = target_net.state_dict()
            # policy_net_state_dict = policy_net.state_dict()
            # for key in policy_net_state_dict:
            #     target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            # target_net.load_state_dict(target_net_state_dict)
            
            if done:
                for _ in range(100):
                    optimize_model()

                print(game)
                print(f"Episode Score: {game.get_sum()}")
                total_scores.append(game.get_sum())
                best_tile_list.append(game.max_num())
                # if i_episode > 50:
                #   average = sum(total_scores[-50:]) / 50
                #   print(f"50 episode running average: {average}")
                break

            # Update the target network, copying all weights and biases in DQN
            if i_episode % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
                policy_net.train()
            # target_net_state_dict = target_net.state_dict()
            # policy_net_state_dict = policy_net.state_dict()
            # for key in policy_net_state_dict:
            #     target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            # target_net.load_state_dict(target_net_state_dict)
            
            if i_episode % 100 == 0:
                torch.save(policy_net.state_dict(), f'policy_net_new_{NUM_EPISODES}.pth')
                torch.save(target_net.state_dict(), f'target_net_new_{NUM_EPISODES}.pth')

    print('Complete')
    best_tile_list = np.array(best_tile_list)
    total_scores = np.array(total_scores)
    losses = np.array(losses)
    np.save(f'best_tile_new_{NUM_EPISODES}.npy', best_tile_list)
    np.save(f'total_scores_new_{NUM_EPISODES}.npy', total_scores)
    np.save(f'loss.npy', losses)


def main():
    global NUM_EPISODES
    NUM_EPISODES = int(sys.argv[1])
    train()

# %%

if __name__ == '__main__':
    main()
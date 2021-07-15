import argparse
import gym
import numpy as np
from itertools import count
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torch.distributions import Categorical


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log_interval', type=int, default=20, metavar='N',
                    help='interval between training status logs (default: 20)')

args = parser.parse_args()

env = gym.make('LunarLanderContinuous-v2')
env.seed(args.seed)
torch.manual_seed(args.seed)

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IN_FEATURES = 8
ACTIONS_MAIN = [-1.0, 0.025, 0.05, 0.1]
ACTIONS_SECONDARY = [-1.0, 0., 1.0]
NUM_ACTIONS_TOTAL = len(ACTIONS_MAIN) * len(ACTIONS_SECONDARY)
EPISODES_BEFORE_VALIDATION = 20


def action_ind_to_value(action_ind, actions_main, actions_secondary):
    # [-1.0, -0.5]: Left Engine
    # [-0.5, 0.5]: Off
    # [0.5, 1.0]: Right Engine

    main_engine_value = actions_main[int(action_ind) % len(actions_main)]
    secondary_engines_value = actions_secondary[int(action_ind) // len(actions_main)]
    return [main_engine_value, secondary_engines_value]


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(IN_FEATURES, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, NUM_ACTIONS_TOTAL)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = functional.relu(x)
        action_scores = self.affine2(x)
        return functional.softmax(action_scores, dim=1)


policy = Policy().to(device)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode():
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns).to(device)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).to(device).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def run_validation_episode():
    last_observation = env.reset()
    episode_done = False
    episode_reward = 0
    policy.eval()

    with torch.no_grad():
        while not episode_done:
            last_observation_tensor = torch.tensor([last_observation]).to(device)
            probs = policy(last_observation_tensor)
            action_value = action_ind_to_value(probs.max(1)[1], ACTIONS_MAIN, ACTIONS_SECONDARY)
            new_observation, reward, episode_done, _ = env.step(action_value)
            episode_reward += reward
            last_observation = new_observation
            screen = env.render(mode='rgb_array')
            plt.imshow(screen)

    policy.train()
    print('validation, reward: {}'.format(episode_reward))


def policy_gradient_train():
    print('policy gradient algorithm')
    running_reward = 10
    for i_episode in count(1):
        state, ep_reward = env.reset(), 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            action_ind = select_action(state)
            action_value = action_ind_to_value(action_ind, ACTIONS_MAIN, ACTIONS_SECONDARY)
            state, reward, done, _ = env.step(action_value)
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
            run_validation_episode()
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    policy_gradient_train()

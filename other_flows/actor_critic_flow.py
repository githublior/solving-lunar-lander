import argparse
from collections import namedtuple

import gym
import numpy as np
from itertools import count
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torch.distributions import Categorical

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.999, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=40, metavar='N',
                    help='interval between training status logs (default: 10)')

args = parser.parse_args()

env = gym.make('LunarLanderContinuous-v2')
env.seed(args.seed)
torch.manual_seed(args.seed)

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IN_FEATURES = 8
ACTIONS_MAIN = [-1.0, 0.025, 0.05, 0.1, 0.5, 1.0]
ACTIONS_SECONDARY = [-1.0, -0.5, 0., 0.5, 1.0]
NUM_ACTIONS_TOTAL = len(ACTIONS_MAIN) * len(ACTIONS_SECONDARY)
EPISODES_BEFORE_VALIDATION = 20


def action_ind_to_value(action_ind, actions_main, actions_secondary):
    # [-1.0, -0.5]: Left Engine
    # [-0.5, 0.5]: Off
    # [0.5, 1.0]: Right Engine

    main_engine_value = actions_main[int(action_ind) % len(actions_main)]
    secondary_engines_value = actions_secondary[int(action_ind) // len(actions_main)]
    return [main_engine_value, secondary_engines_value]


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(IN_FEATURES, 128)

        # actor's layer
        self.action_head = nn.Linear(128, NUM_ACTIONS_TOTAL)

        # critic's layer
        self.value_head = nn.Linear(128, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = functional.relu(self.affine1(x))

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = functional.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values


model = Policy().to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float().to(device)
    probs, state_value = model(state)

    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)

    # and sample an action using the distribution
    action = m.sample()

    # save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # the action to take (left or right)
    return action.item()


def finish_episode():
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + args.gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(functional.smooth_l1_loss(value, torch.tensor([R]).to(device)))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]


def run_validation_episode():
    last_observation = env.reset()
    episode_done = False
    episode_reward = 0
    model.eval()
    with torch.no_grad():
        while not episode_done:
            last_observation_tensor = torch.tensor([last_observation]).to(device)
            probs, _ = model(last_observation_tensor)
            action_value = action_ind_to_value(probs.max(1)[1], ACTIONS_MAIN, ACTIONS_SECONDARY)
            new_observation, reward, episode_done, _ = env.step(action_value)
            episode_reward += reward
            last_observation = new_observation
            screen = env.render(mode='rgb_array')
            plt.imshow(screen)
    model.train()
    print('validation, reward: {}'.format(episode_reward))


def actor_critic_train():
    print('actor critic algorithm')
    running_reward = 10

    # run inifinitely many episodes
    for i_episode in count(1):

        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0

        # for each episode, only run 9999 steps so that we don't
        # infinite loop while learning
        for t in range(1, 10000):

            # select action from policy
            action_ind = select_action(state)
            action_value = action_ind_to_value(action_ind, ACTIONS_MAIN, ACTIONS_SECONDARY)

            # take the action
            state, reward, done, _ = env.step(action_value)

            if args.render:
                env.render()

            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        finish_episode()

        # log results
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
            run_validation_episode()

        # check if we have "solved" the cart pole problem
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    actor_critic_train()

from collections import deque
from itertools import count
import gym
import numpy as np
import torch

from dqn_trainer import DqnTrainer
from hyper_parameters import HyperParameters
from utils import add_noise, init_seeds, distribution, plot_scores, action_chooser


def train_loop(env, agent, action_index_to_value, n_episodes=500, eps_start=0.5, eps_end=0.01, eps_decay=0.994, noise=False):
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        for t in count():
            # if i_episode % 51 == 0:
            #     screen = env.render(mode='rgb_array')
            #     plt.imshow(screen)
            action_ind = agent.act(state, eps)
            action = action_index_to_value(action_ind)
            next_state, reward, done, _ = env.step(action)

            if noise:
                if i_episode == 1 and t == 1:
                    print(' lunar lander with noise')
                next_state = add_noise(next_state)

            agent.step(state, action_ind, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)
        print(
            '\rEpisode {}\tAverage Score: {:.2f}, Steps: {}, Eps: {}'.format(i_episode, np.mean(scores_window), t, eps),
            end="")
        if np.mean(scores_window) > 200:
            print('\nDone: {:d} episodes!\n'.format(i_episode - 100))
        if i_episode % 100 == 0:
            print('\nHigh Score: {:.2f}\tLow Score: {:.2f}'.format(np.max(scores_window), np.min(scores_window)))
    torch.save(agent.q_network_local.state_dict(), 'checkpoint.pth')
    return scores


def main():
    env = gym.make('LunarLanderContinuous-v2')
    init_seeds(env)

    num_main = 3
    num_secondary = 3

    hyper = HyperParameters(
        buffer_size=int(1e5),
        batch_size=64,
        gamma=0.999,
        tau=1e-3,
        lr=1e-3,
        update_every=5,
        h1=128,
        h2=128,
        noise=False
    )

    actions_main = distribution(2 * num_main + 1)[num_main:]
    actions_secondary = distribution(num_secondary)
    num_actions_total = len(actions_main) * len(actions_secondary)

    def action_index_to_value(action_index):
        return action_chooser(action_index, actions_main, actions_secondary)

    agent = DqnTrainer(state_size=env.observation_space.shape[0], action_size=num_actions_total, hyper=hyper)
    print("DQN algorithm...")
    scores = train_loop(env, agent, action_index_to_value, noise=hyper.noise)

    plot_scores(scores)


if __name__ == '__main__':
    main()

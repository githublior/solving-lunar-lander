import torch
import numpy as np
import random
from collections import namedtuple, deque
import matplotlib.pyplot as plt


Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])


def init_seeds(env, seed=11):
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)


def action_chooser(action_ind, actions_main, actions_secondary):
    # [-1.0, -0.5]: Left Engine
    # [-0.5, 0.5]: Off
    # [0.5, 1.0]: Right Engine

    main_engine_value = actions_main[int(action_ind) % len(actions_main)]
    secondary_engines_value = actions_secondary[int(action_ind) // len(actions_main)]
    return [main_engine_value, secondary_engines_value]


def distribution(n):
    return list(np.arange(n-1).astype('float32') / (n-1) * 2 - 1.0) + [1.0]


def add_noise(state):
    noise_x = np.random.normal(0, 0.05)
    noise_y = np.random.normal(0, 0.05)

    state[0] = state[0] + noise_x
    state[1] = state[1] + noise_y
    return state


def plot_scores(scores):
    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, device):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.device = device

    def add(self, state, action, reward, next_state, done):
        self.memory.append(Experience(state, action, reward, next_state, done))

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)

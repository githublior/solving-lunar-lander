import numpy as np
import random
from model import DuelingQNetwork
import torch
import torch.nn.functional as functional
import torch.optim as optim
from utils import ReplayBuffer


class DqnTrainer:
    def __init__(self, state_size, action_size, hyper):
        self.state_size = state_size
        self.action_size = action_size
        self.device = hyper.device
        self.update_every = hyper.update_every
        self.batch_size = hyper.batch_size
        self.tau = hyper.tau
        self.gamma = hyper.gamma

        self.q_network_local = DuelingQNetwork(state_size, action_size, hyper.h1, hyper.h2).to(self.device)
        self.q_network_target = DuelingQNetwork(state_size, action_size, hyper.h1, hyper.h2).to(self.device)
        for target_param, param in zip(self.q_network_local.parameters(), self.q_network_target.parameters()):
            target_param.data.copy_(param)
            
        self.optimizer = optim.Adam(self.q_network_local.parameters(), lr=hyper.lr)

        self.replay_memory = ReplayBuffer(action_size, hyper.buffer_size, self.batch_size, self.device)

        # Initialize time step (for updating every update_every steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.replay_memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.replay_memory) > self.batch_size:
                experiences = self.replay_memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.q_network_local.eval()
        with torch.no_grad():
            action_values = self.q_network_local(state)
        self.q_network_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        actions = actions.view(actions.size(0), 1)
        dones = dones.view(dones.size(0), 1)
        
        curr_q = self.q_network_local.forward(states).gather(1, actions)
        next_q = self.q_network_target.forward(next_states)
        max_next_q = torch.max(next_q, 1)[0]
        max_next_q = max_next_q.view(max_next_q.size(0), 1)
        expected_q = rewards + (1 - dones) * gamma * max_next_q

        loss = functional.smooth_l1_loss(curr_q, expected_q.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for target_param, param in zip(self.q_network_target.parameters(), self.q_network_local.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

    @staticmethod
    def soft_update(local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

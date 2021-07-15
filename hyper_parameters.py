import torch


class HyperParameters:
    def __init__(
        self,
        buffer_size,    # replay buffer size
        batch_size,     # mini batch size
        gamma,          # discount factor
        tau,            # for soft update of target parameters
        lr,            # learning rate
        update_every,   # how often to update the network
        h1,             # number neurons in first hidden layer
        h2,             # number layer in second hidden layer
        noise=False,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.update_every = update_every
        self.h1 = h1
        self.h2 = h2
        self.noise = noise
        self.device = device


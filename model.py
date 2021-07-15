import torch.nn as nn


class NetOriginal(nn.Module):
    def __init__(self, in_features, out_features, hidden_layers_sizes, p_do=None):
        super(NetOriginal, self).__init__()
        all_layers = []
        for in_size, out_size in zip([in_features] + hidden_layers_sizes[:-1], hidden_layers_sizes):
            all_layers.append(nn.Linear(in_size, out_size))
            if p_do:
                all_layers.append(nn.Dropout(p_do))
            all_layers.append(nn.ReLU())

        all_layers.append(nn.Linear(hidden_layers_sizes[-1], out_features))

        self.all_layers = nn.Sequential(*all_layers)

    def forward(self, x):
        return self.all_layers(x)


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, h1, h2):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, action_size)
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.fc1(state)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size, middle_size1=300, middle_size2=300):
        super(DuelingQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, middle_size1)
        self.relu = nn.ReLU()
        
        self.value_stream = nn.Sequential(
            nn.Linear(middle_size2, middle_size2),
            nn.ReLU(),
            nn.Linear(middle_size2, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(middle_size2, middle_size2),
            nn.ReLU(),
            nn.Linear(middle_size2, action_size)
        )
        
    def forward(self, state):
        x = self.fc1(state)
        x = self.relu(x)
        values = self.value_stream(x)
        advantages = self.advantage_stream(x)
        q_values = values + (advantages - advantages.mean())
        return q_values

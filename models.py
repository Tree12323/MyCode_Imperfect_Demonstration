import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

# input -> action
class Policy(nn.Module):
    # hidden_size:100
    def __init__(self, num_inputs, num_outputs, hidden_size):
        super(Policy, self).__init__()
        # input -> 100
        self.affine1 = nn.Linear(num_inputs, hidden_size)
        # 100 -> 100
        self.affine2 = nn.Linear(hidden_size, hidden_size)
        # 100 -> output
        self.action_mean = nn.Linear(hidden_size, num_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))

        self.saved_actions = []
        self.rewards = []
        self.final_value = 0

    def forward(self, x):
        # x -> 100
        x = F.tanh(self.affine1(x))
        # x -> 100
        x = F.tanh(self.affine2(x))
        # x -> output
        action_mean = self.action_mean(x)
        # std
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        # return mean, log_std, std
        return action_mean, action_log_std, action_std

# input -> 1
class Value(nn.Module):
    def __init__(self, num_inputs, hidden_size):
        super(Value, self).__init__()
        # input -> 100
        self.affine1 = nn.Linear(num_inputs, hidden_size)
        # 100 -> 100
        self.affine2 = nn.Linear(hidden_size, hidden_size)
        # 100 -> 1
        self.value_head = nn.Linear(hidden_size, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        # x -> 100
        x = F.tanh(self.affine1(x))
        # x -> 100
        x = F.tanh(self.affine2(x))

        # x -> 1
        state_values = self.value_head(x)
        return state_values

# input -> 1
class Discriminator(nn.Module):
    def __init__(self, num_inputs, hidden_size):
        super(Discriminator, self).__init__()
        # input -> 100
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        # 100 -> 100
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        # 100 -> 1
        self.linear3 = nn.Linear(hidden_size, 1)
        self.linear3.weight.data.mul_(0.1)
        self.linear3.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.tanh(self.linear1(x))
        x = F.tanh(self.linear2(x))
        #prob = F.sigmoid(self.linear3(x))
        output = self.linear3(x)
        return output

# input -> output
class Generator(nn.Module):
    def __init__(self, num_inputs, hidden_size, num_outputs):
        super(Generator, self).__init__()
        # input -> 100
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        # 100 -> 100
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # 100 -> output
        self.fc3 = nn.Linear(hidden_size, num_outputs)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

# input -> 1
class Classifier(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(Classifier, self).__init__()
        # input -> 40
        self.fc1 = nn.Linear(num_inputs, hidden_dim)
        # 40 -> 40
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # 40 -> 1
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.d1 = nn.Dropout(0.5)
        self.d2 = nn.Dropout(0.5)
        
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = self.d1(torch.tanh(self.fc1(x)))
        x = self.d2(torch.tanh(self.fc2(x)))
        x = self.fc3(x)
        return x

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
# Checking hardware
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Pytorch n layer model
class LeakyMLP(nn.Module):
  def __init__(self, i_size, o_size, h_sizes):
    super(LeakyMLP, self).__init__()
    self.layers = nn.ModuleList()
    #self.layers = []
    self.layers.append(nn.Linear(i_size, h_sizes[0]))
    for i in range(1,len(h_sizes)):
      self.layers.append(nn.Linear(h_sizes[i-1], h_sizes[i]))
    self.layers.append(nn.Linear(h_sizes[-1], o_size))

  def forward(self, x):
    #print(f"\n\n Forward pass: \n")
    #print(x)
    for i in range(len(self.layers)-2):
      x = F.leaky_relu(self.layers[i](x))
      #print(f" Layer {i}: {self.layers[i].weight}")
      #print(f"x {i}: {x}")
    x = torch.sigmoid(self.layers[-2](x))
    x = self.layers[-1](x)
    #print(x)
    return F.softmax(x, dim=1)

  def act(self, state):
    state = torch.from_numpy(state).flatten().float().unsqueeze(0).to(device)
    probs = self.forward(state).cpu()
    #print(probs)
    #input("look ok?")
    m = Categorical(probs)
    action = m.sample()
    #print(action)
    return action.item(), m.log_prob(action)


class LeakyMLP2(nn.Module):
  def __init__(self, i_size, o_size, h_sizes):
    super(LeakyMLP2, self).__init__()
    self.layers = nn.ModuleList()
    #self.layers = []
    self.layers.append(nn.Linear(i_size, h_sizes[0]))
    for i in range(1,len(h_sizes)):
      self.layers.append(nn.Linear(h_sizes[i-1], h_sizes[i]))
    self.layers.append(nn.Linear(h_sizes[-1], o_size))

  def forward(self, x):
    #print(f"\n\n Forward pass: \n")
    #print(x)
    for i in range(len(self.layers)-2):
      x = F.leaky_relu(self.layers[i](x))
      #print(f" Layer {i}: {self.layers[i].weight}")
      #print(f"x {i}: {x}")
    x = torch.sigmoid(self.layers[-2](x))
    x = self.layers[-1](x)
    #print(x)
    return F.softmax(x, dim=1)

  def act(self, state):
    state = torch.from_numpy(state).flatten().float().unsqueeze(0).to(device)
    probs = self.forward(state).cpu()
    #print(probs)
    #input("look ok?")
    m = Categorical(probs)
    action = m.sample()
    #print(action)
    return action.item(), m.log_prob(action)

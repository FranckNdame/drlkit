import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
	
	def __init__(self, state_size, action_size, seed, layer1_units=64, layer2_units=64):
		
		super(DQN, self).__init__()
		self.seed = torch.manual_seed(seed)
		self.fc1 = nn.Linear(state_size, layer1_units)
		self.fc2 = nn.Linear(layer1_units, layer2_units)
		self.fc3 = nn.Linear(layer2_units, action_size)
		
	def foward(self, state):
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		return self.fc3(x)
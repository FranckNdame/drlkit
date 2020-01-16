import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvDQN(nn.Module):
	"""Agent Model."""
	
	def __init__(self, state_size, action_size, seed, layer1_units=64, layer2_units=64):
		"""Initialize parameters and build model.
				Params
				======
					state_size (int): Dimension of each state
					action_size (int): Dimension of each action
					seed (int): Random seed
					layer1_units (int): Number of nodes in first hidden layer
					layer2_units (int): Number of nodes in second hidden layer
		"""
		
		super(DQN, self).__init__()
		self.seed = torch.manual_seed(seed)
		self.conv1 = nn.Conv1d(state_size, layer1_units, (4,4))
		self.conv2 = nn.Conv1d(layer1_units, layer2_units, (2,2))
		self.fc3 = nn.Linear(layer2_units, action_size)
		
	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		return self.fc3(x)

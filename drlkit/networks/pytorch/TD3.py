import torch
import torch.nn as nn
import torch.nn.Functional as F

class Actor(nn.Module):
	""" Neural Network for the Actor Model """
	
	def __init__(self, state_size, action_size, max_action, seed=0, layer1_units= 400, layer2_units=300):
		"""Initialize parameters and build model.
				Params
				======
					state_size (int): Dimension of each state
					action_size (int): Dimension of each action
					seed (int): Random seed
					layer1_units (int): Number of nodes in first hidden layer
					layer2_units (int): Number of nodes in second hidden layer
		"""
		
		super(Actor, self).__init__()
		self.layer_1 = nn.Linear(state_size, layer1_units)
		self.layer_2 = nn.Linear(layer1_units, layer2_units)
		self.layer_3 = nn. Linear(layer2_units, action_size)
		self.max_action = max_action
		
	def forward(self, x):
		x = F.relu(self.layer_1(x))
		x = F.relu(self.layer_2(x))
		x = self.max_action * torch.tanh(self.layer_3(x))
		return x
		


		
		
		
	
		
		
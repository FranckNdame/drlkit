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
		

class Critic(nn.Module):
	""" Neural Network for the Critic Model """
	
	def __init__(self, state_size, action_size, seed=0, first_layer_units=400, second_layer_units=300):
		"""Initialize parameters and build model.
				Params
				======
					state_size (int): Dimension of each state
					action_size (int): Dimension of each action
					seed (int): Random seed
					layer1_units (int): Number of nodes in first hidden layer
					layer2_units (int): Number of nodes in second hidden layer
		"""
		
		super(Critic, self).__init__()
		# First Critic Network
		self.layer_1 = nn.Linear(state_size + action_size, first_layer_units)
		self.layer_2 = nn.Linear(first_layer_units, second_layer_units)
		self.layer_3 = nn.Linear(second_layer_units, 1)
		
		# Second Critic Network
		self.layer_4 = nn.Linear(state_size + action_size, first_layer_units)
		self.layer_5 = nn.Linear(first_layer_units, second_layer_units)
		self.layer_6 = nn.Linear(second_layer_units, 1)
		
		
	def forward(self, x, u):
		xu = torch.cat([x, u], 1)
		
		# Forward Propagation on the first Critic neural Network
		x1 = F.relu(self.layer_1(xu))
		x1 = F.relu(self.layer_2(x1))
		x1 = self.layer_3(x1)
		
		x2 = F.relu(self.layer_4(xu))
		x2 = F.relu(self.layer_5(x2))
		x2 = self.layer_6(x2)
		
		return x1, x2
		
	def Q1(self, x, u):
		# Concatebate x and u
		xu = torch.cat([x,u], 1)
		x1 = F.relu(self.layer_1(xu))
		x1 = F.relu(self.layer_2(x1))
		x1 = self.layer_3(x1)
		return x1
		
	def Q2(self, x, u):
		xu = torch.cat([x, u], 1)
		x2 = F.relu(self.layer_4(xu))
		x2 = F.relu(self.layer_5(xu))
		x2 = self.layer_6(xu)
		return x2
			
	
		
		
from __future__ import absolute_import
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


import sys
import os
import random
sys.path.append("..")

from drlkit.utils.memory import Memory
from drlkit.utils import preprocess as pr
from drlkit.networks.pytorch.DQN import DQN



# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TorchAgent(object):
	def __init__(
		self, state_size, action_size, seed,
		buffer_size = 1_000_000, batch_size=64, gamma=0.99,
		tau=1e-3, lr=5e-4, update_every=4
	):
		"""
		Initialize a Torch Agent object.
				
		Params
		======
			state_size (int): dimension of each state
			action_size (int): dimension of each action
			seed (int): random seed
			
		Hyper Params
		======
			buffer_size (int): replay buffer size
			batch_size (int): batch size
			gamma (float): discount rate
			tau float: soft update of target parameters
			lr (float): learning rate
			update_every (int): network update frequency
		"""
		
		# Params
		self.state_size = state_size
		self.action_size = action_size
		self.seed = random.seed(seed)
		
		# Hyperparameters
		self.gamma = gamma
		self.tau = tau
		self.update_every = update_every
		
		# Policy Network
		self.policy_network = DQN(state_size, action_size, seed).to(device)
	
		# Target Network
		self.target_network =  DQN(state_size, action_size, seed).to(device)
		# Optimizer
		self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
		
		# Experience replay
		self.memory = Memory(action_size, seed, buffer_size, batch_size)
		
		self.time_step = 0
		
		
	def step(self, state, action, reward, next_state, done):
		# Save experience to memory
		self.memory.save(state, action, reward, next_state, done)
		
		# learn i.e update weights at self.update_every
		self.time_step = (self.time_step + 1) % self.update_every
		if not self.time_step:
			# If the memory can provide sample
			if self.memory.can_provide:
				experience = self.memory.sample()
				self.learn(experience, self.gamma)
				
		
	def act(self, state, eps=0.):
		"""Returns an action for the given state following the current policy
		
		Params
		======
			state (list): current state
			eps (float): exploration rate
		
		"""
		# Unsqueeze: TODO
		
		state = torch.from_numpy(state).float().unsqueeze(0).to(device)
		self.policy_network.eval()
		with torch.no_grad():
			action_values = self.policy_network(state)
		self.policy_network.train()
		
		if random.random() > eps:
			# Select greedy action
			return np.argmax(action_values.cpu().data.numpy())
		else:
			# select stochastic action
			return random.choice(np.arange(self.action_size))
			
		
	def learn(self, experiences, gamma):
		"""Update weights and bias using batch of experience tuples.
		
		Params
		======
			experiences (Tuple[torch.Tensor]): tuple of state, action, reward, next_state, done tuples
			gamma (float): Discount rate
		"""
		
		states, actions, rewards, next_states, dones = experiences
		
		# Get the predicted action for the next state from the target network
		Q_targets_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
		
		# Get Q target for current state
		Q_target = rewards + (gamma * Q_targets_next * (1- dones))
		
		# Get expected Q from policy model
		Q_expected = self.policy_network(states).gather(1, actions)
		
		# Compute loss
		"""
		loss = (sqrt[target_network^2 - policy_network^2])
		"""
		loss = F.mse_loss(Q_expected, Q_target)
		
		# Minimize loss
		self.optimizer.zero_grad() # Clear gradients
		loss.backward()
		self.optimizer.step() # Perform a single optimization step
		# Update target network
		self.soft_update(self.policy_network, self.target_network, self.tau)
		
	def soft_update(self, policy_model, target_model, tau):
		"""Soft update model parameters.
		θ_target = τ*θ_policy+ (1 - τ)*θ_target

		Params
		======
			policy_model: weights will be copied from
			target_model: weights will be copied to
			tau (float): interpolation parameter 
		"""
		for target_param, policy_param in zip(target_model.parameters(), policy_model.parameters()):
			target_param.data.copy_(tau*policy_param.data + (1.0-tau)*target_param.data)
				
				
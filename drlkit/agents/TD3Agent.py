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
# Selecting the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Building the whole Training Process into a class

class TD3(object):
	
	def __init__(self, state_space, action_space, max_action. seed=0):
		self.actor = Actor(state_space, action_space, max_action).to(device)
		self.actor_target = Actor(state_space, action_space, max_action).to(device)
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
		self.critic = Critic(state_space, action_space).to(device)
		self.critic_target = Critic(state_space, action_space).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
		self.max_action = max_action
		
		self.memory = Memory(action_size, seed, buffer_size, batch_size)

	def select_action(self, state):
		state = torch.Tensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()

	def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
		
		for it in range(iterations):
			
			# Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
			batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = self.memory.sample(batch_size)
			state = torch.Tensor(batch_states).to(device)
			next_state = torch.Tensor(batch_next_states).to(device)
			action = torch.Tensor(batch_actions).to(device)
			reward = torch.Tensor(batch_rewards).to(device)
			done = torch.Tensor(batch_dones).to(device)
			
			# Step 5: From the next state s’, the Actor target plays the next action a’
			next_action = self.actor_target(next_state)
			
			# Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
			noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
			noise = noise.clamp(-noise_clip, noise_clip)
			next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
			
			# Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			
			# Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
			target_Q = torch.min(target_Q1, target_Q2)
			
			# Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
			target_Q = reward + ((1 - done) * discount * target_Q).detach()
			
			# Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
			current_Q1, current_Q2 = self.critic(state, action)
			
			# Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
			
			# Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_optimizer.step()
			
			# Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
			if it % policy_freq == 0:
				actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
				self.actor_optimizer.zero_grad()
				actor_loss.backward()
				self.actor_optimizer.step()
				
				# Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
				for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
					target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
				
				# Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
				for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
					target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
	
	# Making a save method to save a trained model
	def save(self, filename, directory):
		torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
		torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
	
	# Making a load method to load a pre-trained model
	def load(self, filename, directory):
		self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
		self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
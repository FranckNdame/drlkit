import gym
import os
from collections import deque
import sys
import torch
import numpy as np
sys.path.append("..")


class EnvironmentWrapper(object):
	def __init__(
		self, name, max_ts=1_000, 
		eps_start=1.0, eps_min =0.01, eps_decay=0.995,
		seed=0, print_info=False
	):
		self.env_name = name
		env = gym.make(name)
		env.seed(seed)
		self.env = env
		
		self.scores = []
		self.scores_window = deque(maxlen=100)
		self.eps = eps_start
		
		
		self.max_ts = max_ts
		self.eps_min = eps_min
		self.eps_decay = eps_decay
		self.print_info = print_info
		
		self.done = False
		
		
	def fit(self, agent, n_episodes):
		self.agent = agent
		self.n_episodes = n_episodes
		for i_episode in range(1, n_episodes+1):
			state = self.env.reset()
			score = 0
			for t in range(self.max_ts):
				action = agent.act(state, self.eps)
				next_state, reward, self.done, info = self.env.step(action)
				if self.print_info: print(info)
				agent.step(state, action, reward, next_state, self.done)
				state = next_state
				score += reward
				if self.done:
					break
			# After the episode
			self.scores_window.append(score) # push recent score
			self.scores.append(score) # save recent score
			self.eps = max(self.eps_min, self.eps*self.eps_decay) # decrease exploration rate
			if self.monitor_progress(i_episode):
				break
			
	def monitor_progress(self, episode):
		print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(self.scores_window)), end="")
		if not episode % 100:
			print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(self.scores_window)))
		if np.mean(self.scores_window)>=-100.0:
				print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode, np.mean(self.scores_window)))
				filename = f'episode-{episode}[score={round(np.mean(self.scores_window))}].pth'
				self.save_model(f"models/{self.env_name}/", filename, self.agent.target_network.state_dict())
				return True
		
				
		
	def save_model(self, dir, filename, file):
		"""Save model
				
		Params
		======
			dir (string): directory
			filename (string): filename
			file (dictionary): model parameters		
		"""
		if not os.path.exists(dir):
			os.makedirs(dir)
			print("Directory created")
		torch.save(file, dir+filename)
		print("Model saved!")
			
	def list_models(self, dir=""):
		pass
		
	def load_model(self, agent, env, elapsed_episodes):
		self.agent = agent
		agent.target_network.load_state_dict(torch.load('models/LunarLander-v2/episode-254-avg_score--80.0.pth'))
		
		
	def play(self, num_episodes=10, max_ts=200, trained=True):
		for i in range(1,num_episodes+1):
			state = self.env.reset()
			for ts in range(max_ts):
				action = self.agent.act(state) if trained else self.env.action_space.sample()
				self.env.render()
				state, reward, done, _ = self.env.step(action)
				if done:
					break 
					
		self.env.close()
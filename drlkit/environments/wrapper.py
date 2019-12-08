import gym
import os
from collections import deque
import sys
import torch
import numpy as np
sys.path.append("..")
from drlkit import models
from drlkit.utils.exceptions import AgentMissing

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
		self.agent = None
		
		self.scores = []
		self.scores_window = deque(maxlen=100)
		self.eps = eps_start
		
		
		self.max_ts = max_ts
		self.eps_min = eps_min
		self.eps_decay = eps_decay
		self.print_info = print_info
		
		self.done = False
		self.best_score = 0
		
		
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
			self.monitor_progress(i_episode)
		print("==== Training complete! =====")
		print(f"# Episodes: {n_episodes} || score: {np.mean(self.scores_window)}")
			
	def monitor_progress(self, episode):
		print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(self.scores_window)), end="")
		if not episode % 100:
			print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(self.scores_window)))
		if np.mean(self.scores_window)>=200.0:
				if np.mean(self.scores_window) > self.best_score:
					print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode, np.mean(self.scores_window)))
					filename = f'{self.env_name}-{episode}.pth'
					self.best_score = np.mean(self.scores_window)
					self.save_model(filename, self.agent.target_network.state_dict())
			
		
				
		
	def save_model(self, filename, file):
		"""Save model
				
		Params
		======
			dir (string): directory
			filename (string): filename
			file (dictionary): model parameters		
		"""

		path = "./models/"
		if not os.path.exists(path):
			os.makedirs(path)
			print("Directory created")
		torch.save(file, path+filename)
		print(f"Model saved! @ {path+filename}")
			
	def list_models(self):
		path = path = "./models/"
		if not os.path.exists(path):
			print(f"No model found!")
		else:
			lst = os.listdir(path)
			print(f"Models Available")
			print("===========================")
			i = 1
			for item in lst:
				if item != "__init__.py":
					print(f"{i} - {item}") 
					i += 1
		
		
	def load_model(self, agent, env_name, elapsed_episodes):
		self.agent = agent
		path = os.path.dirname(models.__file__)
		path += f"/{env_name}/episode-{elapsed_episodes}.pth"
		if not os.path.exists(path):
			print(f"No such model saved! @ {path}")
			return
		if torch.cuda.is_available():	
			self.agent.target_network.load_state_dict(torch.load(path))
		else:
			self.agent.target_network.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
		print("Model Loaded!")
		
	def load_prebuilt_model(self, agent, path):
			self.agent = agent
			if not os.path.exists(path):
				print(f"No such model saved!  @ {path}")
				return
			if torch.cuda.is_available():	
				self.agent.policy_network.load_state_dict(torch.load(path))
			else:
				self.agent.policy_network.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
				
				
			print("Model Loaded!")
		
		
	def play(self, num_episodes=10, max_ts=200, trained=True):
		if not self.agent:
			raise AgentMissing()
		for i in range(1,num_episodes+1):
			state = self.env.reset()
			for ts in range(max_ts):
				action = self.agent.act(state) if trained else self.env.action_space.sample()
				self.env.render()
				state, reward, done, _ = self.env.step(action)
				if done:
					break 
					
		self.env.close()
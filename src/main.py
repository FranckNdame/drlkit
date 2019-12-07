import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import os

from agents.TorchAgent import TorchAgent
from utils.plot import Plot
#  pip3 install box2d-py
ENV_NAME = "LunarLander-v2"
env = gym.make(ENV_NAME)
env.seed(0)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

agent = TorchAgent(state_size=8, action_size=env.action_space.n, seed=0)
state = env.reset()
done = False
print_info = False

def play(n_episodes, time_steps=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
	"""Train agent
	
	Params
	======
		n_episodes (int): maximum number of training episodes
		time_steps (int): maximum number of timesteps per episode
		eps_start (float): starting value of epsilon
		eps_end (float): minimum value of epsilon
		eps_decay (float): decay rate of epsilon		
	"""
	
	scores = []
	scores_window = deque(maxlen=100)
	eps = eps_start
	for i_episode in range(1, n_episodes+1):
		state = env.reset()
		score = 0
		for t in range(time_steps):
			action = agent.act(state, eps)
			next_state, reward, done, info = env.step(action)
			if print_info:
				print(info)
			agent.step(state, action, reward, next_state, done)
			state = next_state
			score += reward
			if done:
				break
		scores_window.append(score) # push recent score
		scores.append(score) # save recent score
		eps = max(eps_end, eps_decay*eps) # decrease exploration rate
		
		print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
		if not i_episode % 100:
			print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
		if np.mean(scores_window)>=-80.0:
				print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
				filename = f'episode-{i_episode}-avg_score-{round(np.mean(scores_window))}.pth'
				saveModel(f"models/{ENV_NAME}/", filename, agent.target_network.state_dict())
				break
	return scores
	
def saveModel(dir, filename, file):
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
	
	
scores = play(500)
Plot.basic_plot(np.arange(len(scores)), scores, xlabel='Episode #', ylabel='Score')



##### WATCH TRAINED AGENT
#
#
##load the weights from file
#agent.target_network.load_state_dict(torch.load('models/checkpoint.pth'))
#print("Dumb agent")
#for i in range(10):
#	state = env.reset()
#	for j in range(200):
#		env.render()
#		state, reward, done, _ = env.step(env.action_space.sample())
#		if done:
#			break 
#			
#env.close()
#
#print("Smart agent")
#for i in range(10):
#	state = env.reset()
#	for j in range(200):
#		action = agent.act(state)
#		env.render()
#		state, reward, done, _ = env.step(action)
#		if done:
#			break 
#			
#env.close()
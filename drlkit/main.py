import numpy as np
from agents.TorchAgent import TorchAgent
from utils.plot import Plot
from environments.wrapper import EnvironmentWrapper
import os

ENV_NAME = "LunarLander-v2"
env = EnvironmentWrapper(ENV_NAME)
agent = TorchAgent(state_size=8, action_size=env.env.action_space.n, seed=0)

env.list_models(ENV_NAME)
## Train the agent
#env.fit(agent, n_episodes=1000)
#
## See the results
#Plot.basic_plot(np.arange(len(env.scores)), env.scores, xlabel='Episode #', ylabel='Score')


## Play untrained agent
#env.load_model(agent, env="LunarLander", elapsed_episodes=3000)
#env.play(num_episodes=10, trained=False)
#
## Play trained agent
#env.play(num_episodes=10, trained=True)

#path = f"./models/{ENV_NAME}"
#if not os.path.exists(path):
#	print(f"No model saved for {ENV_NAME}")
#else:
#	lst = os.listdir(path)
#	print(f"Models for {ENV_NAME}")
#	print("==================")
#	i = 1
#	for item in lst:
#		if item != "__init__.py":
#			print(f"{i} - {item}") 
#			i += 1
import numpy as np
from agents.TorchAgent import TorchAgent
from utils.plot import Plot
from environments.wrapper import EnvironmentWrapper
import os
import models

ENV_NAME = "LunarLander-v2"
env = EnvironmentWrapper(ENV_NAME, max_ts=5000)
agent = TorchAgent(state_size=8, action_size=env.env.action_space.n, seed=0)

## Train the agent
# env.fit(agent, n_episodes=2_000)
#
## See the results
#Plot.basic_plot(np.arange(len(env.scores)), env.scores, xlabel='Episode #', ylabel='Score')


env.load_prebuilt_model(agent, "./models/LunarLander-v2-4477.pth")
env.play(num_episodes=15, trained=False)




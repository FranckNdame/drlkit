import numpy as np
from drlkit import TorchAgent, Plot, EnvironmentWrapper

ENV_NAME = "LunarLander-v2"
env = EnvironmentWrapper(ENV_NAME)
agent = TorchAgent(state_size=8, action_size=env.env.action_space.n, seed=0)

# Train the agent
env.fit(agent, n_episodes=1000)

# See the results
Plot.basic_plot(np.arange(len(env.scores)), env.scores, xlabel='Episode #', ylabel='Score')


# Play untrained agent
env.load_model(agent, env="LunarLander", elapsed_episodes=3000)
env.play(num_episodes=10, trained=False)

# Play trained agent
env.play(num_episodes=10, trained=True)
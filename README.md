<p align="center">
 <img src="https://raw.githubusercontent.com/FranckNdame/drlkit/master/images/drl-kit-banner.png" width=70% alt="Gitter">
</p>

--------------------------------------------------------------------------------
<br/>
<p align="center">
   <a>
      <img src="https://img.shields.io/badge/python-3.6+-blue.svg" alt="Gitter">
   </a>
   <a>
      <img src="https://github.com/FranckNdame/drlkit/blob/master/images/torchbadge.svg" alt="Pytorch">
   </a>
   <a>
      <img src="https://camo.githubusercontent.com/7ce7d8e78ad8ddab3bea83bb9b98128528bae110/68747470733a2f2f616c65656e34322e6769746875622e696f2f6261646765732f7372632f74656e736f72666c6f772e737667" alt="Gitter">
   </a>
   <a href="https://opensource.org/licenses/MIT">
      <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="Gitter">
   </a>
   <a href="https://opensource.org/licenses/MIT">
      <img src="https://app.fossa.com/api/projects/git%2Bgithub.com%2Fwau%2Fkeras-rl2.svg?type=shield" alt="Gitter">
   </a>
  </p>

<br/>
<p align="center">
 A High Level Python Deep Reinforcement Learning library.<br> 
 Great for beginners,  prototyping and quickly comparing algorithms
</p>

<br/><br/>
<p align="center">
 <img src="images/environments.gif" width=95% alt="Environments">
</p>


## Installation

Run the following to install:

```python
pip install drlkit
```

## Usage

```python
import numpy as np
from agents.TorchAgent import TorchAgent
from utils.plot import Plot
from environments.wrapper import EnvironmentWrapper

ENV_NAME = "LunarLander-v2"
env = EnvironmentWrapper(ENV_NAME)
agent = TorchAgent(state_size=8, action_size=env.env.action_space.n, seed=0)

# Train the agent
env.fit(agent, n_episodes=1000)

# See the results
Plot.basic_plot(np.arange(len(env.scores)), env.scores, xlabel='Episode #', ylabel='Score')

# Play trained agent
env.play(num_episodes=10, trained=True)
```
It is as simple as that! 🤯

### Loading a model
```python
ENV_NAME = "LunarLander-v2"
env = EnvironmentWrapper(ENV_NAME)
agent = TorchAgent(state_size=8, action_size=env.env.action_space.n, seed=0)

env.load_model(agent, "./models/LunarLander-v2-4477.pth")
env.play(num_episodes=10)
```

### Play untrained agent
```python
env.play(num_episodes=10, trained=False)
```
<br>
<p align="center">
 <img src="images/Untrained-Agent.gif" width=50% alt="Environments">
</p>


### Play trained agent (4477 episodes, 3 hours)
```python
env.play(num_episodes=10, trained=True)
```
<br>
<p align="center">
 <img src="images/Trained-Agent.gif" width=50% alt="Environments">
</p>

## Tested Environments

## Algorithms

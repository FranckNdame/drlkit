# drlkit
A High Level Python Deep Reinforcement Learning library. Great for beginners,  prototyping and quickly comparing algorithms

## Installation

Run the following to install:

```python
pip install drlkit
```

## Usage

```python
import drlkit as rl
from drlkit import DQN, EnvManager

# Default imports
import gym
import numpy as np
import tensorflow as tf

env = gym.make("Your-selected-environment")
input = rl.preprocess(env)
model = DQN(num_hidden_layers=3)
model.play(render=True, render_every=10)

# Watch agent
model.watch()
```

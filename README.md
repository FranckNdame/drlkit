<p align="center">
 <img src="https://raw.githubusercontent.com/FranckNdame/drlkit/master/images/drl-kit-banner.png" width=70% alt="Gitter">
</p>

--------------------------------------------------------------------------------
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

A High Level Python Deep Reinforcement Learning library. Great for beginners,  prototyping and quickly comparing algorithms
<br><br>
<p align="center">
 <img src="https://i.ibb.co/QYDKTrv/environments.gif" width=95% alt="Environments">
</p>

# UNDER CONSTRUCTION!
Do not use yet!

| System | 3.5 | 3.6 | 3.7 |
| :---: | :---: | :---: | :--: |
| Linux CPU | [![Build Status](https://ci.pytorch.org/jenkins/job/pytorch-master/badge/icon)](https://ci.pytorch.org/jenkins/job/pytorch-master/) | [![Build Status](https://ci.pytorch.org/jenkins/job/pytorch-master/badge/icon)](https://ci.pytorch.org/jenkins/job/pytorch-master/) | <center>—</center> |
| Linux GPU | [![Build Status](https://ci.pytorch.org/jenkins/job/pytorch-master/badge/icon)](https://ci.pytorch.org/jenkins/job/pytorch-master/) | [![Build Status](https://ci.pytorch.org/jenkins/job/pytorch-master/badge/icon)](https://ci.pytorch.org/jenkins/job/pytorch-master/) | <center>—</center> |
| Windows CPU / GPU | <center>—</center> | [![Build Status](https://ci.pytorch.org/jenkins/job/pytorch-builds/job/pytorch-win-ws2016-cuda9-cudnn7-py3-trigger/badge/icon)](https://ci.pytorch.org/jenkins/job/pytorch-builds/job/pytorch-win-ws2016-cuda9-cudnn7-py3-trigger/) |  <center>—</center> |
| Linux (ppc64le) CPU | [![Build Status](https://powerci.osuosl.org/job/pytorch-master-nightly-py2-linux-ppc64le/badge/icon)](https://powerci.osuosl.org/job/pytorch-master-nightly-py2-linux-ppc64le/) | — | [![Build Status](https://powerci.osuosl.org/job/pytorch-master-nightly-py3-linux-ppc64le/badge/icon)](https://powerci.osuosl.org/job/pytorch-master-nightly-py3-linux-ppc64le/) |
| Linux (ppc64le) GPU | [![Build Status](https://powerci.osuosl.org/job/pytorch-linux-cuda9-cudnn7-py2-mpi-build-test-gpu/badge/icon)](https://powerci.osuosl.org/job/pytorch-linux-cuda9-cudnn7-py2-mpi-build-test-gpu/) | — | [![Build Status](https://powerci.osuosl.org/job/pytorch-linux-cuda92-cudnn7-py3-mpi-build-test-gpu/badge/icon)](https://powerci.osuosl.org/job/pytorch-linux-cuda92-cudnn7-py3-mpi-build-test-gpu/) |


## Installation

Run the following to install:

```python
pip install drlkit
```

## Usage

```python
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
```

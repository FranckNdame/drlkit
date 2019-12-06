import torch
import torch.nn.functional as F
import torch.optim as optim

import sys
sys.path.append("..")
from models.pytorch.DQN import DQN
from utils.experience_replay import Memory
from utils import preprocess as pr


class TorchAgent(object):
	def __init__(self):
		pass
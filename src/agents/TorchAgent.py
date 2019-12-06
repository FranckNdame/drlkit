import torch
import torch.nn.functional as F
import torch.optim as optim

print(torch.cuda.is_available() )
import sys
sys.path.append("..")
from models.pytorch.DQN import DQN
from utils.memory import Memory
from utils import preprocess as pr

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TorchAgent(object):
	def __init__(
		self, state_size, action_size, seed,
		buffer_size = 1_000_000, batch_size=64, gamma=0.9,
		tau=1e-3, lr=5e-4, update_every=4
	):
		"""
		Initialize a Torch Agent object.
				
				Params
				======
					state_size (int): dimension of each state
					action_size (int): dimension of each action
					seed (int): random seed
					
				Hyper Params
				======
					buffer_size (int): replay buffer size
					batch_size (int): batch size
					gamma (float): discount rate
					tau float: soft update of target parameters
					lr (float): learning rate
					update_every (int): network update frequency
		"""
		
		# Params
		self.state_size = state_size
		self.action_size = action_size
		self.seed = seed
		
		# Hyperparameters
		self.buffer_size = buffer_size
		self.batch_size = batch_size
		self.gamma = gamma
		self.tau = tau
		#self.lr = lr
		self.update_every = update_every
		
		# Policy Network
		self.policy_network = DQN(state_size, action_size, seed).to(device)
	
		# Target Network
		self.target_network =  DQN(state_size, action_size, seed).to(device)
		# Optimizer
		self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
		
		# Experience replay
		# TODO
		
		self.time_step = 0
		
		
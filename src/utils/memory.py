from collections import deque, namedtuple
import numpy as np
import random

class Memory(object):
    """Replay memory buffer to store experience tuples."""
    def __init__(self, action_size, buffer_size=1_000_000, batch_size=64, seed):
        """Initialize a ReplayBuffer object.
        
        Params
        ======
            action_size (int): size of each action
            buffer_size (int): size of memory buffer
            batch_size (int): size of each training minibatch
            seed (int): random seed
        """
        self.action_size = action_size
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = seed
        
        self.experience = namedtuple("Experience", field_names=["states", "actions", "reward", "next_state", "done"])

    def save(self, state, action, reward, next_state, done):
        """Save new experience to buffer"""
        experience = self.experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self):
        """Sample a batch of random experiences from memory"""
        samples = random.sample(self.buffer, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([sample.state for sample in samples if sample is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([sample.action for sample in samples if sample is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([sample.reward for sample in samples if sample is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([sample.next_state for sample in samples if sample is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([sample.done for sample in samples if sample is not None]).astype(np.uint8)).float().to(device)
          
        return (states, actions, rewards, next_states, dones)
        
        
    @property
    def can_provide(self):
        """Check if the buffer can provide samples"""
        return len(self.buffer) >= self.batch_size

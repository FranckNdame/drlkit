from collections import deque
import numpy as np

import random
class Memory(object):
    def __init__(self, buffer_size=1_000_000):
        self.buffer = deque(maxlen=buffer_size)

    def save(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        sample_size = min(len(self.buffer), batch_size)
        samples = random.choices(self.buffer, k=sample_size)
        return map(list, zip(*samples))

#    def sample(self, batch_size):
#        if batch_size > len(self.buffer):
#            return
#        index = np.random.choice(
#            np.arange(len(self.buffer)),
#            size = batch_size,
#            replace = False
#        )
#        return [self.buffer[i] for i in index]

    def can_provide(self, batch_size):
        return len(self.buffer) >= batch_size

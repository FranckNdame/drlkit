from collections import deque

class Memory(object):
    def __init__(self, buffer_size=1_000_000):
        self.buffer = deque(maxlen=buffer_size)

    def save(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            return
        index = np.random.choice(
            np.arrange(len(self.buffer)),
            size = batch_size,
            replace = False
        )

import numpy as np
from collections import deque
import random

class ReplayBuffer(object):
    """
    Reply Buffer
    """
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque()
        self.count = 0

    # buffer에 transition 추가
    def add_buffer(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)

        # buffer에 transition이 가득 찼을 경우, 기존 transition을 삭제하고 
        # 새로운 transition을 add
        if self.count < self.buffer_size:
            self.buffer.append(transition)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(transition)

    
    def sample_batch(self, batch_size):
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
            #batch size보다 buffer의 크기가 작으면 count의 크기만큼 batch 추출
        else:
            batch = random.sample(self.buffer, batch_size)
        states = np.asarray([i[0] for i in batch])
        actions = np.asarray([i[1] for i in batch])
        rewards = np.asarray([i[2] for i in batch])
        next_states = np.asarray([i[3] for i in batch])
        dones = np.asarray([i[4] for i in batch])
        return states, actions, rewards, next_states, dones


    ## Current buffer occupation
    def buffer_count(self):
        return self.count


    ## Clear buffer
    def clear_buffer(self):
        self.buffer = deque()
        self.count = 0
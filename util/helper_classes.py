from collections import namedtuple
import math
import torch
import torch.nn as nn
import random

from .helper_functions import sequence_masks

class ParallelTable(nn.Module):
    def __init__(self, model1, model2):
        super(ParallelTable,self).__init__()
        self.layer1 = model1
        self.layer2 = model2
        
    def forward(self, x):
        y = (self.layer1(x[0]), self.layer2(x[1]))
        return y


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class MBLoader:
    def __init__(self, inputs, batch_size, user_stats = None):
        self.inputs = inputs
        self.batch_ind = [0, 0, 0]
        self.batch_size = batch_size
        get_num_batches = lambda inp : int(math.ceil(inp[0].size(0)/float(self.batch_size)))
        self.num_batches = [get_num_batches(inputs[i]) for i in range(3)]

        self.user_stats = user_stats
        self.max_seq_len = self.inputs[0][0].size(1)

    def load_next_batch(self, split, buzz_info = False):
        end_ind = (self.batch_ind[split]+1) * self.batch_size
        if (self.batch_ind[split]+1) * self.batch_size > self.inputs[split][0].size(0):
            end_ind = self.inputs[split][0].size(0)
            self.batch_ind[split] = 0
        else:
            self.batch_ind[split] = self.batch_ind[split] + 1
            
        start_ind = end_ind - self.batch_size

        mb_X = self.inputs[split][0][start_ind:end_ind]
        mb_y = self.inputs[split][1][start_ind:end_ind]
        mb_len = self.inputs[split][2][start_ind:end_ind]

        all_mask, last_mask = sequence_masks(sequence_length=mb_len, max_len=self.max_seq_len)

        if buzz_info:
            mb_buzzes = self.inputs[split][3][start_ind:end_ind]
            return mb_X, mb_y, mb_len, mb_buzzes, all_mask, last_mask
        else:
            return mb_X, mb_y, mb_len, all_mask, last_mask

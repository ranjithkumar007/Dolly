import math

class MBLoader:
    def __init__(self, inputs, batch_size):
        self.inputs = inputs
        self.batch_ind = {'train' : 0, 'test' : 0, 'val' : 0}
        self.batch_size = batch_size
        get_num_batches = lambda inp : int(math.ceil(inp[0].size(0)/float(self.batch_size)))
        self.num_batches = {'train' : get_num_batches(inputs['train']), 
                            'val' : get_num_batches(inputs['val']), 
                            'test' : get_num_batches(inputs['test'])}

        self.max_seq_len = self.inputs['train'][0].size(1)

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

        if buzz_info:
            mb_buzzes = self.inputs[split][3][start_ind:end_ind]
            return mb_X, mb_y, mb_len, mb_buzzes
        else:
            return mb_X, mb_y, mb_len
        


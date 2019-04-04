import torch
import shutil
import os
from torch.autograd import Variable

def sequence_masks(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand, seq_range_expand == (seq_length_expand - 1)

def load_checkpoint(model, optimizer, logger, filename):
    start_epoch = 0
    min_loss = 99999999999999999
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = checkpoint['logger']
        min_loss = checkpoint['min_loss']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, logger, min_loss

def save_checkpoint(state, is_best, filename):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    if not os.path.isfile(filename):
        os.mknod(filename)

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'checkpoints/model_best.pth.tar')


def load_best_model(model):
    checkpoint = torch.load('checkpoints/model_best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])

    return model

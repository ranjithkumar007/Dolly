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

def get_dir_path(pref):
    if pref == 'content':
        return '../checkpoints/content'
    else:
        return '../checkpoints/buzz'

def load_checkpoint(model, optimizer, logger, filename, pref = 'content'):
    start_epoch = 0
    min_loss = 99999999999999999
    dirpath = get_dir_path(pref)

    if os.path.isfile(dirpath + '/' + filename):
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

def save_checkpoint(state, is_best, filename, pref = 'content'):
    dirpath = get_dir_path(pref)

    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    if not os.path.isfile(dirpath + '/' + filename):
        os.mknod(filename)

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, dirpath + pref + '_model_best.pth')


def load_best_model(model, filename = 'checkpoints/model_best.pth'):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])

    return model


def load_checkpoint_buzz(hyperparams, env, policy_net, optimizer, memory, logger, filename):
    logger = [{'avg_reward' : [], 'avg_buzz_pos' : []} for i in range(3)]
    steps_done = 0
    min_val_reward = 99999999999
    start_game_ind = 0

    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_game_ind = checkpoint['start_game_ind']
        hyperparams = checkpoint['hyperparams']
        policy_net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        env = checkpoint['env']
        memory = checkpoint['memory']
        logger = checkpoint['logger']
        min_val_reward = checkpoint['min_val_reward']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return hyperparams, env, policy_net, optimizer, memory, logger, \
            start_game_ind, steps_done, min_val_reward, optimizer, \
            start_epoch, logger


def plot_from_logger(logger, isbuzz = False):
    keys = list(logger[0].keys())
    split_names = ["train", "val", "test"]
    
    dirpath = None
    if isbuzz:
        dirpath = '../figures/buzz/'
    else:
        dirpath = '../figures/content/'

    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    x = np.arange(1, len(logger[0][keys[0]]) + 1)

    for i in range(len(keys)):
        for j in range(2): # change to 3 to include single dots of test
            plt.plot(x, logger[j][keys[i]])
            plt.xlabel('Number of epochs')

            ylabel = split_names[j] + "_" + keys[i]
            plt.ylabel(ylabel)
            plt.show()
            
            plt.savefig(dirpath + ylabel + '.png')
            plt.close() 

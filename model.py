import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import click
import time
import math
from torch.autograd import Variable

from util.helper_functions import load_checkpoint, save_checkpoint, sequence_masks, load_best_model
from util.game_env import GameEnv

np.random.seed(0)
torch.manual_seed(0)

def train(loader, content_model, buzz_model, criterion, optimizer):
    env = GameEnv(loader, content_model)

    epoch_loss = 0
    correct = 0
    total = 0
    last_total = 0
    last_correct = 0
    end = time.time()
    batch_size = content_model.batch_size
    num_batch = loader.num_batches[0] # split = 0 for train
    max_seq_len = loader.max_seq_len
    state_dim = env.state_dim

    with click.progressbar(range(num_batch)) as batch_indexes:
        for batch_i in batch_indexes:
            mb_X, mb_y, mb_len, mb_buzzes, all_mask, last_mask = loader.load_next_batch(0, True)
            all_mask = all_mask.flatten().float()
            last_mask = last_mask.flatten().float()

            outputs = content_model(mb_X, mb_len)
            outputs = outputs.view(batch_size, max_seq_len, -1)

           	sa_values = None
            with torch.no_grad():
	            state_feat = torch.zeros((batch_size, max_seq_len, state_dim))
	            player_buzz_pos = []
	            player_cor = []
	            

				ans_prob = outputs.softmax(dim = 2)
				temp = torch.zeros((batch_size, max_seq_len, 1))
				ans_prob_copy = torch.cat((temp, ans_prob), 2)[:,:,:-1]
				prob_cat = torch.cat((ans_prob, ans_prob_copy), 2)

				state_feat[:, :, :prob_cat.size(2)] = prob_cat

	            for i, mb_buzz in enumerate(mb_buzzes):
	            	ind = prob_cat.size(2)
	            	state_feat[i, :, ind] = torch.arange(1, max_seq_len + 1)
	            	ind += 1

	            	player_buzz = mb_buzz[torch.randint(len(mb_buzz), (1, 1))]

	            	temp = torch.ones([1, player_buzz[1] - 1]).long()
	            	other = torch.ones(max_seq_len) - temp

	            	if player_buzz[2]:
	            		state_feat[i, :, ind] = other
	            	else:
	            		state_feat[i, :, ind + 1] = other

	            	ind += 2

	            	state_feat[i, :(player_buzz[1] - 1), ind] = temp
	            	ind += 1
					
					state_feat[i, :, ind] = self.user_stats[self.player_id]['overall_acc']
					state_feat[i, :, ind + 1] = self.user_stats[self.player_id]['mean_frac']
					state_feat[i, :, ind + 2] = self.user_stats[self.player_id]['total_ques']

					ind += 3

				sa_values = buzz_model(state_feat)
            mb_y = mb_y.view(-1, 1).repeat(1, max_seq_len)

           	weighted_loss = 0
			for ts in range(max_seq_len):
				ts_w = max(0.1, min(1, sa_values[:, ts, 1] - sa_values[:, ts, 0]))
            	loss = criterion(outputs[:, ts, :], mb_y[:, ts])
            	weighted_loss += loss
 
			outputs = outputs.view(batch_size * max_seq_len, -1)
			_, predicted_labels = torch.max(outputs, dim = 2)
            mb_y = mb_y.flatten()

            matched = (predicted_labels == mb_y).float().cpu()
            correct += (all_mask * matched).sum()
            total += all_mask.sum()

            last_correct += (last_mask * matched).sum()
            last_total += last_mask.sum()

            epoch_loss += float(weighted_loss)

            optimizer.zero_grad()
            weighted_loss.backward()

            torch.nn.utils.clip_grad_norm_(content_model.parameters(), 10)
            optimizer.step()

    
    avg_loss = epoch_loss / batch_size
    avg_acc = correct / total
    last_acc = last_correct / last_total

    return avg_loss, avg_acc, last_acc


def validate(loader, model, criterion, split):

    with torch.no_grad():

        epoch_loss = 0
        correct = 0
        total = 0
        last_total = 0
        last_correct = 0

        end = time.time()
        batch_size = model.batch_size
        num_batch = loader.num_batches[split]
        max_seq_len = loader.max_seq_len
        
        with click.progressbar(range(num_batch)) as batch_indexes:
            for batch_i in batch_indexes:
                mb_X, mb_y, mb_len, mb_buzzes, all_mask, last_mask = loader.load_next_batch(0, True)
            all_mask = all_mask.flatten().float()
            last_mask = last_mask.flatten().float()

            outputs = content_model(mb_X, mb_len)
            outputs = outputs.view(batch_size, max_seq_len, -1)

            state_feat = torch.zeros((batch_size, max_seq_len, state_dim))
            player_buzz_pos = []
            player_cor = []
            

			ans_prob = outputs.softmax(dim = 2)
			temp = torch.zeros((batch_size, max_seq_len, 1))
			ans_prob_copy = torch.cat((temp, ans_prob), 2)[:,:,:-1]
			prob_cat = torch.cat((ans_prob, ans_prob_copy), 2)

			state_feat[:, :, :prob_cat.size(2)] = prob_cat

            for i, mb_buzz in enumerate(mb_buzzes):
            	ind = prob_cat.size(2)
            	state_feat[i, :, ind] = torch.arange(1, max_seq_len + 1)
            	ind += 1

            	player_buzz = mb_buzz[torch.randint(len(mb_buzz), (1, 1))]

            	temp = torch.ones([1, player_buzz[1] - 1]).long()
            	other = torch.ones(max_seq_len) - temp

            	if player_buzz[2]:
            		state_feat[i, :, ind] = other
            	else:
            		state_feat[i, :, ind + 1] = other

            	ind += 2

            	state_feat[i, :(player_buzz[1] - 1), ind] = temp
            	ind += 1
				
				state_feat[i, :, ind] = self.user_stats[self.player_id]['overall_acc']
				state_feat[i, :, ind + 1] = self.user_stats[self.player_id]['mean_frac']
				state_feat[i, :, ind + 2] = self.user_stats[self.player_id]['total_ques']

				ind += 3

			sa_values = buzz_model(state_feat)
            mb_y = mb_y.view(-1, 1).repeat(1, max_seq_len)

           	weighted_loss = 0
			for ts in range(max_seq_len):
				ts_w = max(0.1, min(1, sa_values[:, ts, 1] - sa_values[:, ts, 0]))
            	loss = criterion(outputs[:, ts, :], mb_y[:, ts])
            	weighted_loss += loss
 
			outputs = outputs.view(batch_size * max_seq_len, -1)
			_, predicted_labels = torch.max(outputs, dim = 2)
            mb_y = mb_y.flatten()

            matched = (predicted_labels == mb_y).float().cpu()
            correct += (all_mask * matched).sum()
            total += all_mask.sum()

            last_correct += (last_mask * matched).sum()
            last_total += last_mask.sum()

            epoch_loss += float(weighted_loss)


        avg_loss = epoch_loss / batch_size
        avg_acc = correct / total
        last_acc = last_correct / last_total

    return avg_loss, avg_acc, last_acc


def run(loader, content_model, buzz_model, criterion, optimizer, early_stopping, early_stopping_interval, checkpoint_file, num_epochs, restore = True):
    logger = [{'loss' : [], 'last_acc' : [], 'avg_acc' : []} for i in range(3)]

    start_epoch = 1
    min_loss = 99999999999999999
    ntrial = 0

    if restore:
        content_model, optimizer, start_epoch, logger, min_loss = load_checkpoint(content_model, optimizer, logger, checkpoint_file)

    for epoch in range(start_epoch, num_epochs + 1):
        
        train_loss, avg_acc, last_acc = train(loader, model, criterion, optimizer)
        logger[0]['loss'].append(train_loss)
        logger[0]['last_acc'].append(last_acc)
        logger[0]['avg_acc'].append(avg_acc)
        
        print('On training set : Epoch:  %d | Loss: %.4f | avg_acc : %.2f | last_acc : %.2f' 
          %(epoch, train_loss, avg_acc, last_acc)) 

        val_loss, avg_acc, last_acc = validate(loader, model, criterion, split = 1)
        logger[1]['loss'].append(val_loss)
        logger[1]['last_acc'].append(last_acc)
        logger[1]['avg_acc'].append(avg_acc)
        
        is_best = False
        if val_loss < min_loss:
            min_loss = val_loss
            is_best = True
            ntrial = 0
            print("Best Model Found")
        else:
            ntrial = ntrial + 1
            if early_stopping and ntrial >= early_stopping_interval:
                print("Early stopping! Validation error didn't improve since last " + str(ntrial) + " epochs")
                break

        print('On Validation set : Epoch:  %d | Loss: %.4f | avg_acc : %.2f | last_acc : %.2f' 
          %(epoch, val_loss, avg_acc, last_acc)) 

        save_checkpoint({'epoch': epoch + 1,
            'content_state_dict': content_model.state_dict(),
            'logger': logger,
            'min_loss' : min_loss,
            'optimizer' : optimizer.state_dict()}, is_best, checkpoint_file)

    model = load_best_model(model, filename = 'checkpoints/best_model.pth')
    test_loss, avg_acc, last_acc = validate(loader, model, criterion, split = 2)
    print('On Test set(Best from validation set)  Loss: %.4f | avg_acc : %.2f | last_acc : %.2f' 
          %(test_loss, avg_acc, last_acc))

    logger[2]['loss'].append(test_loss)
    logger[2]['last_acc'].append(last_acc)
    logger[2]['avg_acc'].append(avg_acc)
    
    return logger

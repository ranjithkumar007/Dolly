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

np.random.seed(0)
torch.manual_seed(0)

class QA_RNN(nn.Module):
    def __init__(self, batch_size, n_steps, n_layers, hidden_size, n_outputs, embed_mat, dropout, non_trainable = True, disable_cuda = False):
        super(QA_RNN, self).__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_layers = n_layers
        self.n_outputs = n_outputs
        self.non_trainable = non_trainable
        self.embed_mat = embed_mat
        self.dropout = dropout

        self.device = None
        if not disable_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        self.__build_model()

    def __build_model(self):
        # embedding layer
        num_embeddings, embedding_dim = self.embed_mat.shape
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight': self.embed_mat})

        if self.non_trainable:
            emb_layer.weight.requires_grad = False

        self.word_embedding = emb_layer
        
        self.gru = nn.GRU(embedding_dim, self.hidden_size, self.n_layers, batch_first = True) 

        self.FC = nn.Linear(self.hidden_size, self.n_outputs)

    
    def init_hidden(self):
        # (num_layers, batch_size, n_neurons)
        hidden = torch.randn(self.n_layers, self.batch_size, self.hidden_size)
        hidden = hidden.cuda()
        return Variable(hidden)
        
    def forward(self, X, X_lengths):
        # transforms X to dimensions: n_steps X batch_size X n_inputs
        # X = X.permute(1, 0, 2) 

        self.hidden = self.init_hidden()
        X = self.word_embedding(X) # bs X seq_len X embedding_dim

        X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True)
        X = self.gru(X, self.hidden)[0] # bs X seq_len X hidden_size
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True, total_length = self.n_steps)

        # project to tag space
        X = X.contiguous()
        X = X.view(-1, X.shape[2])

        X = self.FC(X) # BS * seq_len X n_outputs

        return X


def train(loader, model, criterion, optimizer):
    epoch_loss = 0
    correct = 0
    total = 0
    last_total = 0
    last_correct = 0
    end = time.time()
    batch_size = model.batch_size
    num_batch = loader.num_batches[0] # split = 0 for train
    max_seq_len = loader.max_seq_len

    with click.progressbar(range(num_batch)) as batch_indexes:
        for batch_i in batch_indexes:
            mb_X, mb_y, mb_len, all_mask, last_mask = loader.load_next_batch(0, False)
            all_mask = all_mask.flatten().float()
            last_mask = last_mask.flatten().float()

            mb_y = mb_y.view(-1, 1).repeat(1, max_seq_len).flatten()
            outputs = model(mb_X, mb_len)
            # loss = criterion(outputs, mb_y)
            losses = criterion(outputs, mb_y)
            loss = (all_mask.cuda() * losses).sum()

            _, predicted_labels = torch.max(outputs, dim = 1)
            
            matched = (predicted_labels == mb_y).float().cpu()
            correct += (all_mask * matched).sum()
            total += all_mask.sum()

            last_correct += (last_mask * matched).sum()
            last_total += last_mask.sum()

            epoch_loss += float(loss)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

    
    avg_loss = epoch_loss / total
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
                mb_X, mb_y, mb_len, all_mask, last_mask = loader.load_next_batch(split, False) 
                all_mask = all_mask.flatten().float()
                last_mask = last_mask.flatten().float()

                mb_y = mb_y.view(-1, 1).repeat(1, max_seq_len).flatten()
                outputs = model(mb_X, mb_len)
                losses = criterion(outputs, mb_y)
                loss = (all_mask.cuda() * losses).sum()

                _, predicted_labels = torch.max(outputs, dim = 1)
                
                matched = (predicted_labels == mb_y).float().cpu()
                correct += (all_mask * matched).sum()
                total += all_mask.sum()

                last_correct += (last_mask * matched).sum()
                last_total += last_mask.sum()

                epoch_loss += float(loss)
        
        avg_loss = epoch_loss / total
        avg_acc = correct / total
        last_acc = last_correct / last_total

    return avg_loss, avg_acc, last_acc


def run(loader, model, criterion, optimizer, early_stopping, early_stopping_interval, checkpoint_file, num_epochs, restore = True):
    logger = [{'loss' : [], 'last_acc' : [], 'avg_acc' : []} for i in range(3)]

    start_epoch = 1
    min_loss = 99999999999999999
    ntrial = 0

    if restore:
        model, optimizer, start_epoch, logger, min_loss = load_checkpoint(model, optimizer, logger, checkpoint_file)

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
            'state_dict': model.state_dict(),
            'logger': logger,
            'min_loss' : min_loss,
            'optimizer' : optimizer.state_dict()}, is_best, checkpoint_file)

    model = load_best_model(model, filename = 'checkpoints/content/best_model.pth')
    test_loss, avg_acc, last_acc = validate(loader, model, criterion, split = 2)
    print('On Test set(Best from validation set)  Loss: %.4f | avg_acc : %.2f | last_acc : %.2f' 
          %(test_loss, avg_acc, last_acc))

    logger[2]['loss'].append(test_loss)
    logger[2]['last_acc'].append(last_acc)
    logger[2]['avg_acc'].append(avg_acc)
    
    return logger

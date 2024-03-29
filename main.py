import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
import click
import pickle
import torch.backends.cudnn as cudnn

from content_model import QA_RNN
from buzz_model import DDQNQuizBowlPlayer

from util.helper_functions import load_checkpoint_policy, load_checkpoint
from util.helper_classes import MBLoader

np.random.seed(0)
torch.manual_seed(0)

@click.command()
@click.option('--model_name', default="buzz_RL", help='Name of model.',show_default=True)
@click.option('--data_dir', default="data/", help='Path to dataset file containing questions.')
@click.option('--content_model_path', default="checkpoints/content/best_model.pth", help='Path of checkpoint_file of best content model')
@click.option('--buzz_model_path', default="checkpoints/buzz/policy.pth", help='Path of checkpoint_file of best buzz model')
@click.option('--checkpoint_file', default="checkpoints/dolly/checkpoint.pth", help='Path of checkpoint_file of dolly')
@click.option('--batch_size', default=64, help="Batch size.",show_default=True)
@click.option('--num_layers', default=1, help="Number of RNN layers.",show_default=True)
@click.option('--learning_rate', default=0.001, help="LR",show_default=True)
@click.option('--state_size', default=128, help="RNN state size.",show_default=True)
@click.option('--dropout', default=0.0, help="keep_prob for droupout.",show_default=True)
@click.option('--val_interval', default=1, help='validation interval for early stopping. ',show_default=True)
@click.option('--num_epochs', default=50, help='Number of iteration to train.',show_default=True)
@click.option('--train_embeddings', default=False, is_flag=True, help='train word embeddings.',show_default=True)
@click.option('--disable_cuda', default=False, is_flag=True, help='run on gpu or not',show_default=True)
@click.option('--restore', default=False, is_flag=True, help='restore previous model',show_default=True)
@click.option('--early_stopping', default=True, is_flag=True, help='early stopping on validation error.',show_default=True)
@click.option('--early_stopping_interval', default=15, help='early stopping on validation error.',show_default=True)
def main(model_name,data_dir,batch_size,num_layers,learning_rate, state_size,dropout,val_interval,early_stopping_interval,num_epochs,train_embeddings,early_stopping,disable_cuda,content_model_path,buzz_model_path,checkpoint_file,restore):
    preprocessed_file = os.path.join(data_dir,"preprocessed_data.npz")
    nf = np.load(preprocessed_file)
    train_X,train_y,train_seq_len,\
    train_buzzes,\
    test_X,test_y,test_seq_len,\
    test_buzzes,\
    val_X,val_y,val_seq_len,\
    val_buzzes,\
    embd_mat = nf["train_X"],nf["train_y"],nf["train_seq_len"],\
        nf["train_buzzes"],\
        nf["test_X"],nf["test_y"],nf["test_seq_len"],\
        nf["test_buzzes"],\
        nf["val_X"],nf["val_y"],nf["val_seq_len"],\
        nf["val_buzzes"],\
        nf["embd_mat"]

    print(list(map(lambda x:x.shape  ,[train_X,train_y,train_seq_len,train_buzzes])))
    print(list(map(lambda x:x.shape  ,[test_X,test_y,test_seq_len,test_buzzes])))
    print(list(map(lambda x:x.shape  ,[val_X,val_y,val_seq_len,val_buzzes])))

    in_file = os.path.join(data_dir,"mapping_opp.pkl")
    with open(in_file,"rb") as handle:
        user_features = pickle.load(handle)
        user_features = user_features[0]
   
    num_ans = len(set(train_y)|set(test_y)|set(val_y))
    print("#Answers :",num_ans)

    model_name = model_name+"_"+str(train_X.shape[0])+"_"+str(val_X.shape[0])+"_"+str(test_X.shape[0])+"_"+str(batch_size)+"_"+str(dropout)

    train_X = torch.from_numpy(train_X)
    train_y = torch.from_numpy(train_y)
    train_seq_len = torch.from_numpy(train_seq_len)
    val_X = torch.from_numpy(val_X)
    val_y = torch.from_numpy(val_y)
    val_seq_len = torch.from_numpy(val_seq_len)
    test_X = torch.from_numpy(test_X)
    test_y = torch.from_numpy(test_y)
    test_seq_len = torch.from_numpy(test_seq_len)
    embd_mat = torch.from_numpy(embd_mat)#.cuda()


    content_model = QA_RNN(batch_size, train_X.size(1), num_layers, state_size, num_ans + 1, embd_mat, non_trainable = True, disable_cuda = disable_cuda)
    buzz_model = DDQNQuizBowlPlayer(inp_state_dim, opp_state_dim, n_actions)

    inputs = [(train_X,train_y,train_seq_len), 
                (val_X,val_y,val_seq_len), 
                (test_X,test_y,test_seq_len)]

    loader = MBLoader(inputs, batch_size)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, content_model.parameters()), lr=learning_rate)

    content_model = load_best_model(content_model, content_model_path)
    buzz_model = load_checkpoint_policy(buzz_model, buzz_model_path)

    if not disable_cuda:
        torch.backends.cudnn.enabled = True
        cudnn.benchmark = True
        model.cuda()
        criterion = criterion.cuda()
        train_X = train_X.cuda()
        # train_seq_len = train_seq_len.cpu()
        train_y = train_y.cuda()
        test_X = test_X.cuda()
        test_y = test_y.cuda()
        # test_seq_len = test_seq_len.cpu()
        val_X = val_X.cuda()
        val_y = val_y.cuda()
        # val_seq_len = val_seq_len.cpu()
        


    print(optimizer)
    print(criterion)
    # print(next(model.parameters()).is_cuda)

    logger = run(loader, content_model, buzz_model, criterion, optimizer, early_stopping, early_stopping_interval, checkpoint_file = checkpoint_file, num_epochs = num_epochs, restore = restore)

    plot_from_logger(logger)

if __name__ == '__main__':
    main()


import warnings
warnings.filterwarnings("ignore")

import click
import pandas as pd
import os
from nltk.tokenize import word_tokenize
import numpy as np
import pickle

from convert_raw_to_csv import convert2csv

PAD = '<eos>'
UNK = '<unk>'

def convert_to_ids(data,word2id,ans2id):
    X = []
    Y = []
    buzzes = []
    with click.progressbar(range(data.shape[0]),label="Converting to ids") as row_indexes:
        for i in row_indexes:
            que = data.iloc[i][0]
            ans = data.iloc[i][1]

            que_words = word_tokenize(que)            
            X_i = [word2id[word] if word in word2id.keys() else word2id[UNK] for word in que_words]

            X.append(X_i)
            Y.append(ans2id[ans])

            buzzes.append([list(x.split('-')) for x in data.iloc[i][3].split('|')])

    seq_len = [len(X_i) for X_i in X]   

    X = np.array(X)
    Y = np.array(Y)
    seq_len = np.array(seq_len)
    buzzes = np.array(buzzes)

    perm_idx = seq_len.argsort()[::-1]
    X = X[perm_idx]
    Y = Y[perm_idx]
    seq_len = seq_len[perm_idx]
    buzzes = buzzes[perm_idx]

    return X,Y,seq_len,buzzes

def build_vocab(data):
    vocab = {}
    for dt in data:
        for sent in dt:
            words = sent.split(' ')
            for word in words:
                if word not in vocab:
                    vocab[word] = 0
                vocab[word] = vocab[word] + 1

    return vocab

def filter_vocab(vocab, cutoff):
    filtered = []
    for w, cnt in vocab.items():
        if cnt < cutoff:
            filtered.append(w)

    for w in filtered:
        del vocab[w]

    return vocab

def create_mapping(vocab, spl_toks):
    mapping = {}
    sz = 0

    for w in vocab.keys():
        mapping[w] = sz
        sz = sz + 1

    for w in spl_toks:
        mapping[w] = sz
        sz = sz + 1

    return sz, mapping

def create_dicts(train,test,val,glove_file,glove_dim):
    #ans2id
    answers = set(train[1])|set(test[1])|set(val[1])
    print("#Answers :",len(answers))

    id2ans = dict(enumerate(answers))
    ans2id = dict([(y,x) for (x,y) in id2ans.items()])

    vocab = build_vocab([train[0], test[0], val[0]])
    print("Init vocab size : ", len(vocab))
    vocab = filter_vocab(vocab, cutoff = 5)
    
    vocab_size, vocab_mapping = create_mapping(vocab, [UNK, PAD])

    print("Final vocab size : ",vocab_size)

    # print("Reading",glove_file)
    emb_vec = dict()
    with open(glove_file) as handle, click.progressbar(length=vocab_size) as pbar:
        for line in handle:
            ls = line.split()
            if ls[0] in vocab_mapping:
                emb_vec[ls[0]] = np.array(ls[-glove_dim:],dtype=np.float32)
                pbar.update(1)


    print("Words found : ",len(emb_vec),'/',vocab_size)
    
    # id2word = dict([ (x+1,y) for (x,y) in enumerate(wordsToVec.keys())])
    id2word = dict([ (x,y) for (x,y) in enumerate(vocab_mapping)])
    word2id = dict([(y,x) for (x,y) in id2word.items()])

    embd_mat = np.zeros((len(word2id),glove_dim))
    num_unks = 0

    for i, w in id2word.items():
        if w in emb_vec:
            embd_mat[i,:] = emb_vec[w]
        else:
            embd_mat[i, :] = np.random.uniform(-0.05, 0.05, glove_dim)
            num_unks = num_unks + 1

    print("num of unknowsn : ", num_unks)
    print("embd_mat.shape :",embd_mat.shape)

    return ans2id,word2id,embd_mat

def pad_sequences(X,max_seq_len, word2id):
    return np.array([ X_i + [word2id[PAD]]*(max_seq_len-len(X_i)) for X_i in X ])

@click.command()
@click.option('--data_dir', default="data/", help='Path to dataset directory.')
@click.option('--raw_file', default="buzz_data.txt", help='Path to rawfile')
@click.option('--glove_file', default="../old_tf_code/datasets/glove.6B.300d.txt", help='Glove file path.')
@click.option('--convert_to_csv', is_flag = True, default=False, help='convert raw data to csv format.',show_default=True)
@click.option('--glove_dim', default=300, help='Dimention of glove vectors.',show_default=True)
def main(data_dir,glove_file,glove_dim,raw_file, convert_to_csv):
    if convert_to_csv:
        buzz = False
        if "buzz" in raw_file:
            buzz = True
        convert2csv(data_dir, raw_file, buzz)

    # loading data
    def load_data(filename):
        data_file = os.path.join(data_dir,filename)
        return pd.read_csv(data_file,header=None)

    def load_data2(filename):
        data_file = os.path.join("data/",filename)
        return pd.read_csv(data_file,header=None)

    train,test,val = map(load_data,["train.csv","test.csv","val.csv"])
    print(train.shape,test.shape,val.shape)

    ans2id,word2id,embd_mat = create_dicts(train,test,val,glove_file,glove_dim)    
    

    # # converting to vectors
    print("Creating data_X and data_Y.")
    train_X,train_Y,train_seq_len,train_buzzes = convert_to_ids(train,word2id,ans2id)
    print(list(map(lambda x:x.shape  ,[train_X,train_Y,train_seq_len,train_buzzes])))

    test_X,test_Y,test_seq_len,test_buzzes = convert_to_ids(test,word2id,ans2id)
    print(list(map(lambda x:x.shape  ,[test_X,test_Y,test_buzzes])))

    val_X,val_Y,val_seq_len,val_buzzes = convert_to_ids(val,word2id,ans2id)
    print(list(map(lambda x:x.shape  ,[val_X,val_Y,val_seq_len,val_buzzes])))

    
    max_seq_len = max(np.max(train_seq_len),np.max(test_seq_len),np.max(val_seq_len))
    print("max_seq_len :",max_seq_len)

    print("Padding")
    train_X = pad_sequences(train_X,max_seq_len, word2id)
    test_X = pad_sequences(test_X,max_seq_len, word2id)
    val_X = pad_sequences(val_X,max_seq_len, word2id)
    print(train_X.shape)
    print(test_X.shape)
    print(val_X.shape)

    # Output file
    out_file = os.path.join(data_dir,"preprocessed_data.npz")
    np.savez(out_file,

        train_X=train_X,train_y=train_Y,train_seq_len=train_seq_len,
        train_buzzes=train_buzzes,
        
        test_X=test_X,test_y=test_Y,test_seq_len=test_seq_len,
        test_buzzes=test_buzzes,

        val_X=val_X,val_y=val_Y,val_seq_len=val_seq_len,
        val_buzzes=val_buzzes,

        embd_mat=embd_mat,
        padding_index = word2id[PAD],
        unk_index = word2id[UNK])

    # #output dicts
    out_file = os.path.join(data_dir,"mapping.pkl")
    with open(out_file,"wb") as handle:
        pickle.dump([word2id,ans2id], handle, protocol=pickle.HIGHEST_PROTOCOL)
    

    # user stats
    buzzes = pd.read_csv(data_dir + "/user_stats_on_train.csv")
    print("#Buzzes :",buzzes.shape[0])
    buzzes.head(10)

    print("Human Accuracy : ",buzzes["correct"].mean())
    users = buzzes["user"].value_counts()
    print("#users : ", users.shape[0])
    user_features = {}
    grouped_df = buzzes[["user","frac_ans_position","correct"]].groupby("user")
    for user_id, det in grouped_df:
        assert(user_id not in user_features)
        overall_acc = det.mean()["correct"]
        mean_frac = det["frac_ans_position"].mean()
        user_features[user_id]  = {"overall_acc" : overall_acc,
                                    "mean_frac" : mean_frac,
                                    "total_ques":  det.shape[0]}

    out_file = os.path.join(data_dir,"mapping_opp.pkl")
    with open(out_file,"wb") as handle:
        pickle.dump([user_features], handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__=="__main__":
    main()

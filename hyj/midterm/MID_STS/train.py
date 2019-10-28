import numpy as np
import os
import copy
import torch
import torch.nn as nn
import torch.functional
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models
import torchvision.transforms as transforms
import torchvision.utils
from torch.nn.utils import clip_grad_norm_
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import pylab
import keras
from  keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, Lambda, Dropout
from keras.layers import LSTM, Bidirectional
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import Input, layers, optimizers, regularizers

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)
'''
class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))
    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids
'''

def load_sts_data(path):
    """
    Load STS benchmark data.
    """
    assert os.path.exists(path)
    genres, sent1, sent2, labels, scores = [], [], [], [], []
    for line in open(path,'r',encoding='utf-8'):
        genre = line.split('\t')[0].strip()
        filename = line.split('\t')[1].strip()
        year = line.split('\t')[2].strip()
        other = line.split('\t')[3].strip()
        score = line.split('\t')[4].strip()
        s1 = line.split('\t')[5].strip()
        words = s1.split()
        s2 = line.split('\t')[6].strip()
        words = s2.split()
        label = float(score)
        genres.append(genre)
        sent1.append(s1)
        sent2.append(s2)
        labels.append(label)
        scores.append(score)
    labels = (np.asarray(labels)).flatten()
    return genres, sent1, sent2, labels, scores


def tokenizing_and_padding(data_path, tokenizer, max_len):
    """
    Tokenizing and padding sentences to max length.
    """
    genres, sent1, sent2, labels, scores = load_sts_data(data_path)
    sent1_seq = tokenizer.texts_to_sequences(sent1)
    sent2_seq = tokenizer.texts_to_sequences(sent2)
    sent1_seq_pad = pad_sequences(sent1_seq, max_len)
    sent2_seq_pad = pad_sequences(sent2_seq, max_len)
    print('Shape of data tensor:', sent1_seq_pad.shape)
    print('Shape of label tensor:', labels.shape)

    return sent1_seq_pad, sent2_seq_pad, labels


def encode_labels(labels):
    """
    Encode labels Tai et al., 2015
    """
    labels_to_probs = []
    for label in labels:
        tmp = np.zeros(6, dtype=np.float32)
        if (int(label) + 1 > 5):
            tmp[5] = 1
        else:
            tmp[int(label) + 1] = label - int(label)
            tmp[int(label)] = int(label) - label + 1
        labels_to_probs.append(tmp)

    return np.asarray(labels_to_probs)
def build_emb_matrix(emb_path, vocab_size, word_index):
    """
    Load word embeddings.
    """
    embedding_matrix = np.zeros((vocab_size, 300), dtype='float32')
    with open(emb_path, 'r',encoding='utf-8') as f:
        for i, line in enumerate(f):
            s = line.split(' ')
            if (s[0] in word_index) and (word_index[s[0]] < vocab_size):
                embedding_matrix[word_index[s[0]], :] = np.asarray(s[1:])

    return embedding_matrix

max_words = 114514
max_len = 25
emb_train=True

train_path = 'stsbenchmark/sts-train.csv'
test_path = 'stsbenchmark/sts-test.csv'
dev_path = 'stsbenchmark/sts-dev.csv'
embedding_path = 'data/emb/glove.840B.300d.txt'
#models = ['RNN','LSTM','BILSTM','BERT']
#model = models[0]#RNN
print('Loading data...')
idc = Dictionary()
genres_train,sent1_train,sent2_train,labels_train_,scores_train=load_sts_data(train_path)
genres_dev, sent1_dev, sent2_dev, labels_dev_, scores_dev = load_sts_data(dev_path)

print("Building dictionary...")
text = sent1_train + sent2_train + sent1_dev + sent2_dev
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(text)
word_index = tokenizer.word_index#{'a':1,'the':2,......}

print("Padding and indexing sentences...")#sentence to sequence
sent1_train_seq, sent2_train_seq, labels_train = tokenizing_and_padding(train_path, tokenizer, max_len)
sent1_dev_seq, sent2_dev_seq, labels_dev  = tokenizing_and_padding(dev_path, tokenizer, max_len)

print("Encoding labels...")
train_labels_to_probs = encode_labels(labels_train)
dev_labels_to_probs = encode_labels(labels_dev)

print("Loading embeddings...")
vocab_size = min(max_words, len(word_index)) + 1
embedding_matrix = build_emb_matrix(embedding_path, vocab_size, word_index)
embedding_layer = Embedding(vocab_size, 300,
                                    weights=[embedding_matrix],
                                    input_length=max_len,
                                    trainable=emb_train,
                                    mask_zero=True,
                                    name='VectorLookup')

print(embedding_matrix.shape)

lstm = Bidirectional(LSTM(150, #dropout=dropout_rate, recurrent_dropout=0.1,
                    return_sequences = False, kernel_regularizer=regularizers.l2(1e-4), name='RNN'))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(labels_train)


#classifier = my_RNN()



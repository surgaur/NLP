# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 23:03:03 2020

@author: epocxlabs
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable


class RCNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim,output_dim,num_layers,batch_size):
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn  = nn.RNN(embedding_dim,hidden_dim,num_layers = num_layers,bidirectional = True)
        self.w2 = nn.Linear(hidden_dim * 2 + embedding_dim , hidden_dim)
        self.fc = nn.Linear(hidden_dim,output_dim)
        
    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
        
    def forward(self,x):
        '''
        NOTE In original paper author use Glove but I am using Random Embedding
        
        The idea of the paper "Recurrent Convolutional Neural Networks for Text Classification" is that we pass the embedding vector
		of the text sequences through a bidirectional LSTM and then for each sequence, our final embedding vector is the concatenation of 
		its own GloVe embedding and the left and right contextual embedding which in bidirectional LSTM is same as the corresponding hidden
		state. This final embedding is passed through a linear layer which maps this long concatenated encoding vector back to the hidden_size
		vector. After this step, we use a max pooling layer across all sequences of texts. This converts any varying length text into a fixed
		dimension tensor of size (batch_size, hidden_size) and finally we map this to the output layer.
        '''
        embed = self.embedding(x) # [Batch_size,src len,emb dim]
        encorder_state, hidden = self.rnn(embed)
        ## encorder_state [Batch_size,src len,Hidden_dim*2]
        ## hidden [Forward+Backward,src len,Hidden_dim]
        out = torch.tanh(self.w2(torch.cat((encorder_state,embed),2))) ## [batch_size,seq_length,hidden_dim]
        out = out.permute(0,2,1) ## batch_size,hidden_dim,seq_length
        out = F.max_pool1d(out,out.size()[2]) ## batch_size,hidden_dim,1
        out = self.fc(out.squeeze(2))
        return out
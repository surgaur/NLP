# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 00:34:42 2020

@author: epocxlabs
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class HiearchiclalAttention(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, output_dim,num_layers):
        super().__init__()
        
        self.embed = nn.Embedding( input_dim , emb_dim )
        self.rnn = nn.GRU( emb_dim , hidden_size = enc_hid_dim,bidirectional=True,num_layers = num_layers )
        self.attn = nn.Linear(enc_hid_dim * 2 , enc_hid_dim , bias= True )
        self.contx = nn.Linear(enc_hid_dim,1 ,bias = False)
        self.linear_layer = nn.Linear(enc_hid_dim*2 , output_dim, bias=True )
    
    def forward(self,x):
        x = self.embed(x) # [Batch_size,src len,emb dim]
        encorder_states,hidden = self.rnn(x)  ## encorder_state [Batch_size,src len,Hidden_dim*2]
        inp1 = torch.tanh(self.attn(encorder_states)) # [Batch_size,src len,Hidden_dim]
        inp2 = F.softmax(self.contx(inp1),1) # [Batch_size,src len,1]
        output = (inp2*encorder_states).sum(1) # [Batch_size,Hidden_dim*2]
        out = self.linear_layer(output)
        return out
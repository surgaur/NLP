# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 16:38:50 2020

@author: epocxlabs
"""
### Importing Libraries
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, models
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import albumentations
import pandas as pd
import numpy as np
import io,skimage
import skimage.transform
from torch.utils.data import Dataset, DataLoader
import os,cv2,time
import gc,collections
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
import copy
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from torch import topk
gc.collect()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

### Multi Head Attention 
class MultiHead_Attn(nn.Module):
    def __init__(self,embed_dim,num_heads):        
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        
        self.keys = nn.Linear(self.embed_dim, self.embed_dim  * self.num_heads,bias=False)
        self.queries = nn.Linear(self.embed_dim,self.embed_dim  * self.num_heads,bias=False)
        self.values =  nn.Linear(self.embed_dim,self.embed_dim  * self.num_heads,bias=False)
        self.f_linear = nn.Linear(self.embed_dim  * self.num_heads, self.embed_dim )
        
        
        self.scale = torch.sqrt(torch.FloatTensor([self.embed_dim //num_heads])).to(device)
        
    def forward(self, k , v , q,mask = None):
        '''
        Example : sent = "Bank of the river"
        vocab_size = 4
        embed_dim = 3
        
        expm=torch.randint(0, 4, (4,))
        expm = expm.unsqueeze(0)
        expm,expm.shape
        
        embedding = nn.Embedding(vocab_size,embed_dim)
        embed = embedding(expm)
        embed
        
        num_heads = 1
        keys = nn.Linear(embed_dim,embed_dim * num_heads,bias=False)
        queries = nn.Linear(embed_dim,embed_dim * num_heads,bias=False)
        values = nn.Linear(embed_dim,embed_dim * num_heads,bias=False)
        w_keys = keys(embed)  ## [batch_size,key length,dim]
        w_queries = queries(embed)  ## [batch_size,query length,dim]
        w_values = values(embed)
        
        e = torch.bmm(w_queries,w_keys.permute(0,2,1))
        e = F.softmax(e,1)
        e
        out = torch.matmul(e,w_values)
        out.shape
        
        out shape is torch.Size([1, 4, 3]) and embed shape is torch.Size([1, 4, 3])
                
        '''
        
        
        batch_size = q.shape[0]
        k_len = k.shape[1] ## key_len
        q_len = q.shape[1] ## query_len
        v_len = v.shape[1] ## value_len
        
        
        w_keys =  self.keys(k) ## [batch,key len,embed dim]
        w_queries = self.queries(q) ## [batch,query len,embed dim]
        w_values = self.values(v) ## [batch,value len,embed dim]
        
        w_keys = w_keys.view(batch_size,-1,k_len,self.embed_dim)   ## [batch size,num head,key len, embed dim]
        w_queries = w_queries.view(batch_size,-1,q_len,self.embed_dim)  ## [batch size,num head,query len, embed dim]
        w_values = w_values.view(batch_size,-1,v_len,self.embed_dim)  ## [batch size,num head,value len, embed dim]
        
        '''
         permute change the dimension of keys from [batch size,num head,key len, embed dim] to
         [batch size,num head,embed dim, key len]
        
        '''
        
        energy = torch.matmul (w_queries,w_keys.permute(0,1,3,2)) ## energy [batch size,num head,embed dim,embed dim]
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
            
        energy = energy/self.scale
        energy = F.softmax(energy,-1)
        

        f_energy = torch.matmul(energy,w_values) ## [batch size,num head,embed dim,q len]
        
        f_energy = f_energy.permute(0, 2, 1, 3) ## [batch_size, q length, num heads , embed_dim ] 
        f_energy = f_energy.reshape(batch_size,q_len,-1) ## [batch size,q len ,num head * embed dim]
        out = self.f_linear(f_energy) ## [batch size,q len,embed dim]
        
        
        return out
    
## Point Wise Feed Forward Network   
class Pointwise_FF(nn.Module):
    def __init__(self,embed_dim,expansion):        
        super().__init__()
        
        self.fc_1 = nn.Linear(embed_dim, expansion)
        self.fc_2 = nn.Linear(expansion, embed_dim)
        self.dropout = nn.Dropout(.5)
        
    def forward(self,x):
        
        x = torch.relu(self.fc_1(x))
        x = self.dropout(self.fc_2(x)) ##[batch size, seq len, embed_dim]
        
        return x
    
    
## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self,embed_dim,num_heads,expansion):
        super(TransformerBlock, self).__init__()
        
        self.attn = MultiHead_Attn(embed_dim,num_heads)
        self.norm = nn.LayerNorm(embed_dim) 
        self.ff = Pointwise_FF(embed_dim,expansion)
        
    def forward(self,value,key,query,mask = None):
        
        x = self.attn(value,key,query)
        norm1 = self.norm(x + query)
        norm2 = self.norm(self.ff(norm1) + norm1)
        
        return norm2
    
## Encorder  
class Encorder(nn.Module):
    def __init__(self,src_vocab_size,embed_dim,src_max_length,num_heads,expansion,depth):
        super(Encorder, self).__init__()
        
        self.Embedding_layer = nn.Embedding(src_vocab_size, embed_dim)
        self.positional_Embedding = nn.Embedding(src_max_length, embed_dim)
        
        self.Transformer_layers = TransformerBlock(embed_dim,num_heads,expansion)
        tblocks = []
        
        for i in range(depth):
            tblocks.append(self.Transformer_layers)
            
        self.tblocks = nn.Sequential(*tblocks)
            
        self.dropout = nn.Dropout(.5)
        
    def forward(self,src):
        
        batch_size,seq_len = src.shape
        positions = torch.arange(0,seq_len).expand(batch_size,seq_len).to(device)
        embed = self.dropout(self.Embedding_layer(src) + self.positional_Embedding(positions))
        
        for layer in self.tblocks:            
            x = self.dropout(self.Transformer_layers(embed,embed,embed))
            
        return x
    

## Decorder Block
class DecorderBlock(nn.Module):
    def __init__(self, embed_dim , num_heads , expansion):
        super(DecorderBlock, self).__init__()
        
        self.norm = nn.LayerNorm(embed_dim)
        self.masked_m_h_attn = MultiHead_Attn(embed_dim,num_heads)
        self.transformer = TransformerBlock(embed_dim,num_heads,expansion)
        self.ff = Pointwise_FF(embed_dim,expansion)
        self.dropout = nn.Dropout(.5)
    
    def forward(self,x,value,key,trg_mask):
        
        attn = self.masked_m_h_attn(x , x , x , trg_mask )
        query = self.dropout(self.norm( attn +  x))
        trans = self.transformer( value , key , query )
        norm2 = self.norm( trans + query )
        pff = self.ff( norm2 )
        out = self.norm( pff + norm2 )
        
        return out
    
## Decorder  
class Decoder(nn.Module):
    def __init__(self,trg_vocab_size,embed_dim,trg_max_length,num_heads,expansion,depth):
        super(Decoder, self).__init__()
        
        self.Embedding_layer = nn.Embedding(trg_vocab_size, embed_dim)
        self.positional_Embedding = nn.Embedding(trg_max_length, embed_dim)
        
        self.Transformer_layers = DecorderBlock(embed_dim,num_heads,expansion)
        tblocks = []
        
        for i in range(depth):
            tblocks.append(self.Transformer_layers)
            
        self.tblocks = nn.Sequential(*tblocks)
            
        self.dropout = nn.Dropout(.5)
        self.fc_out = nn.Linear(embed_dim, trg_vocab_size)
        
    def forward(self ,trg , enc_out , trg_mask ):
        
        batch_size,seq_len = trg.shape
        positions = torch.arange(0,seq_len).expand(batch_size,seq_len).to(device)
        embed = self.dropout(self.Embedding_layer(trg) + self.positional_Embedding(positions))
        
        for layer in self.tblocks:            
            x = self.dropout(self.Transformer_layers(embed , enc_out , enc_out , trg_mask))
        
        out = self.dropout(self.fc_out(x))
        return out
  
## Collective seq2seq model with Encorder and decorder
class seq2seq_Transformer(nn.Module):
    def __init__(self,src_vocab_size , trg_vocab_size , embed_dim , src_max_length ,trg_max_length ,num_heads,expansion , depth):
        super(seq2seq_Transformer, self).__init__()
        
        self.enc = Encorder(src_vocab_size , embed_dim , src_max_length , num_heads , expansion , depth)
        self.dec = Decoder(trg_vocab_size , embed_dim , trg_max_length , num_heads , expansion , depth )
        
    def trg_mask_(self,trg):
        Batch_size, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(Batch_size, 1, trg_len, trg_len)
        return trg_mask.to(device)
    
    def forward(self,src , trg ):
        
        trg_mask = self.trg_mask_(trg)
        enc_out = self.enc(src)
        dec = self.dec(trg, enc_out, trg_mask)
        return dec

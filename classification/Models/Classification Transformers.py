# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 19:30:47 2020

@author: epocxlabs
"""

import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MultiHead_Attn(nn.Module):
    def __init__(self,embed_dim,hyper_dim,num_heads):        
        super().__init__()
        self.num_heads = num_heads
        
        self.keys = nn.Linear(embed_dim,hyper_dim * self.num_heads,bias=False)
        self.queries = nn.Linear(embed_dim,hyper_dim * self.num_heads,bias=False)
        self.values = nn.Linear(embed_dim,hyper_dim * self.num_heads,bias=False)
        self.fc = nn.Linear(hyper_dim * self.num_heads,embed_dim)
        
        self.scale = torch.sqrt(torch.FloatTensor([hyper_dim//num_heads])).to(device)
        
    def forward(self,embed):
        
        batch_size,src_len,_ = embed.shape
        
        w_keys = self.keys(embed)  ## [batch_size,key length,dim]
        w_queries = self.queries(embed)  ## [batch_size,query length,dim]
        w_values = self.values(embed)  ## [batch_size,value length,dim]
        
        
        w_keys = w_keys.view(batch_size,self.num_heads,src_len,-1)
        w_queries = w_queries.view(batch_size,self.num_heads,src_len,-1)
        w_values = w_values.view(batch_size,self.num_heads,src_len,-1)
        '''
        w_keys ::: [batch_size , num_heads , key len , hyper_dim]
        w_queries ::: [batch_size , num_heads , query len , hyper_dim]
        w_values ::: [batch_size , num_heads , value len , hyper_dim]   
        
        '''
        e = torch.matmul(w_keys,w_queries.permute(0, 1, 3, 2))
        '''
        e = [batch_size,num head,key len,query len]
        '''
        e = e/self.scale 
        e = torch.softmax(e,-1)
        energy = torch.matmul(e,w_values)
        f_state = energy.permute(0, 2, 1, 3) ## [batch_size, src length, num heads , hyper dim] 
        f_state = f_state.reshape(batch_size,src_len,-1)  ## [batch_size, src length, num heads * hyper dim] 
        
        #f_state = energy.permute(0, 2, 1, 3).contiguous() ## [batch_size, src length, num heads , hyper dim] 
        #f_state = f_state.view(batch_size, -1, hyper_dim)
        
        out = self.fc(f_state) #(batch_size, query_len, embed_size)
        
        #return out,e
        return out
    
    
class Pointwise_FF(nn.Module):
    def __init__(self,embed_dim,rr):        
        super().__init__()
        
        self.fc_1 = nn.Linear(embed_dim, rr)
        self.fc_2 = nn.Linear(rr, embed_dim)
        
    def forward(self,x):
        
        x = torch.relu(self.fc_1(x))
        x = self.fc_2(x) ##[batch size, seq len, embed_dim]
        
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self,embed_dim,hyper_dim,num_heads,rr):
        super(TransformerBlock, self).__init__()
        
        #self.Embedding_layer = Embedding(input_dim,embed_dim).to(device)
        self.self_attn = MultiHead_Attn(embed_dim,hyper_dim,num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.p_ff = Pointwise_FF(embed_dim,rr)
        
        self.norm2 = nn.LayerNorm(embed_dim)
    def forward(self,embed):
                      
        attn = self.self_attn(embed)
                      
        norm1 = self.norm1(attn + embed)
        ff = self.p_ff(norm1)       
        
        out = self.norm2(ff + norm1)
        return out
    
class Transformer(nn.Module):
    def __init__(self, input_dim,embed_dim,hyper_dim,num_heads,rr,depth,output_dim):
        super(Transformer, self).__init__()
        
        self.Embedding_layer = nn.Embedding(input_dim, embed_dim)
        self.transformer = TransformerBlock(embed_dim,hyper_dim,num_heads,rr)
        tblocks = []
        
        for i in range(depth):
            tblocks.append(TransformerBlock(embed_dim,hyper_dim,num_heads,rr))
            
        self.tblocks = nn.Sequential(*tblocks)
        
        self.fc_final = nn.Linear(embed_dim, output_dim )
        
    def forward(self,x):
        
        embed = self.Embedding_layer(x)
        x = self.tblocks(embed)
        x = self.fc_final(x.mean(dim=1))
        
        return x
    
    
    

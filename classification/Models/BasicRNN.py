import torch
import torch.nn as nn
from torch.autograd import Variable



class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim,num_layers,batch_size,bidirectional = True):
        
        '''
    1. nn.Embedding
      Input: batch_size x seq_length
      Output: batch-size x seq_length x embedding_dimension

    2. nn.LSTM
      Input: seq_length x batch_size x input_size (embedding_dimension in this case)
      Output: seq_length x batch_size x hidden_size
      last_hidden_state: batch_size, hidden_size
      last_cell_state: batch, hidden_size
      
    3. nn.Linear
      Input: batch_size x input_size (hidden_size of LSTM in this case or ??)
      Output: batch_size x output_size
      
    4. num_layers is stacked version of RNN/LSTM
        '''
        
        super().__init__()
        if bidirectional:
            direction = 2
        else:
            direction = 1
            
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.RNN  = nn.RNN(embedding_dim,hidden_dim,num_layers =num_layers,bidirectional = bidirectional,
                           batch_first = True)
        self.fc = nn.Linear(hidden_dim * direction , output_dim )
        
    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
        
    def forward(self,x):
        
        #x [src len,batch_size]
        x = self.embedding(x) ## [src len , batch size, emb dim]
        output, hidden = self.RNN(x)
        #assert torch.equal(output[:,-1,:], direction*hidden.squeeze(0))
        '''
        The final hidden state, hidden, has a shape of [num layers * num directions, batch size, hid dim]
        These are ordered: [forward_layer_0, backward_layer_0, forward_layer_1, backward_layer 1,
        ..., forward_layer_n, backward_layer n]. As we want the final (top) layer forward 
        and backward hidden states, we get the top two hidden layers from the first dimension, 
        hidden[-2,:,:] and hidden[-1,:,:], and concatenate them together before passing 
        them to the linear layer 
        '''
        ## ##hidden = [batch size, hid dim * num directions]
        x = self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))        
        return x
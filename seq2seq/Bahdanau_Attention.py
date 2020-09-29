class Encorder(nn.Module):
    def __init__(self, source_input_dim, emb_dim, enc_hid_dim,dec_hid_dim):
        super().__init__()
        
        self.embed = nn.Embedding( source_input_dim , emb_dim )
        self.rnn = nn.GRU( emb_dim , hidden_size=enc_hid_dim,bidirectional=True )
        self.fc = nn.Linear( enc_hid_dim *2 ,dec_hid_dim)
        
    def forward(self,x):
        x = self.embed(x)
        encorder_states,hidden = self.rnn(x)
        '''
        hidden[0:1]==hidden[-2,:,:]
        hidden[1:2]==hidden[-1,:,:]
        hidden  = torch.cat((hidden[0:1], hidden[1:2]), dim = 2)
        
        
        '''
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        hidden = self.fc(hidden)
        
        return encorder_states,hidden



class Attention(nn.Module):
    def __init__(self,enc_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear(3*enc_hid_dim , enc_hid_dim)
        self.v = nn.Linear(enc_hid_dim,1,bias=False)
        
    def forward(self,encorder_state,hidden):
        
        src_len = encorder_state.shape[0]
        batch_size = encorder_state.shape[1]
        
        encorder_state = encorder_state.permute(1,0,2)
        
        hidden = hidden.unsqueeze(1).repeat(1,12,1)
        
        energy = torch.cat((encorder_state,hidden),dim = 2)
        attn = self.v(torch.tanh(self.attn(energy)))
        attn = attn.squeeze(2) ### converting [batch_size,src_len,1] to [batch_size,src_len]
        return F.softmax(attn, dim=1)


class Decorder(nn.Module):
    def __init__(self, target_input_dim, emb_dim, dec_hid_dim,Attention):
        super().__init__()
        
        self.Attention = Attention
        self.embed = nn.Embedding( target_input_dim , emb_dim )
        self.rnn = nn.GRU(dec_hid_dim*3 , dec_hid_dim)
        self.Linear = nn.Linear(dec_hid_dim*3 ,target_input_dim )
    
    def forward(self,x,encorder_state,hidden):
        
        x = x.unsqueeze(1)
        embedded = self.embed(x)
        
        attn = self.Attention(encorder_states,hidden)
        
        weight = torch.bmm(attn.unsqueeze(1), encorder_state.permute(1,0,2))
        
        rnn_input = torch.cat((embedded[0],weight.permute(1,0,2)),dim =2)
        
        output,hidden = self.rnn(rnn_input,hidden.unsqueeze(0))
        assert (output == hidden).all()
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weight.squeeze(0)
        
        prediction = self.Linear(torch.cat((output, weighted, embedded), dim = 1))
        
        return prediction, hidden.squeeze(0)



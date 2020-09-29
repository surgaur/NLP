class Encorder(nn.Module):
    def __init__(self, source_input_dim, emb_dim, enc_hid_dim):
        super().__init__()
        
        self.embedding = nn.Embedding(source_input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim,enc_hid_dim)
        
    def forward(self,x):
        embed = self.embedding(x)
        output,hidden = self.rnn(embed)
        return hidden


class Decorder(nn.Module):
    def __init__(self, target_input_dim,emb_dim,dec_hid_dim):
        super().__init__()
        
        self.embedding = nn.Embedding(target_input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim*2 ,dec_hid_dim )
        self.fc = nn.Linear(dec_hid_dim*3,target_input_dim)
        
    def forward(self,x,hidden,context):
        x = x.unsqueeze(0)
        embed = self.embedding(x)
        embed_context = torch.cat((embed,context),2)
        output,hidden = self.rnn(embed_context,hidden)
        
        output = torch.cat((embed,hidden,res_enc),2)
        output = output.squeeze(0)
        pred=self.fc(output) 
        return pred,hidden
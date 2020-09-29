import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, models
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
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


from torchtext.datasets import Multi30k
from torchtext.data  import BucketIterator,Field
import random
import spacy


SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

import en_core_web_sm
import de_core_news_sm
spacy_eng = en_core_web_sm.load()
spacy_ger = de_core_news_sm.load()


def tokenize_ger(text):
    ### Tokenization of German sentence
    return [tok.text for tok in spacy_ger.tokenizer(text)]
def tokenize_eng(text):
    ### Tokenization of English sentence
    return [tok.text for tok in spacy_eng.tokenizer(text)]


german = Field(tokenize=tokenize_ger,lower= True,init_token = "<sos>",
               eos_token="<eos>")

english = Field(tokenize=tokenize_eng,lower= True,init_token = "<sos>",
               eos_token="<eos>")


train_data , valid_data,test_data = Multi30k.splits(exts = ('.de','.en')
                                                   ,fields = (german,english))

print(f"Number of training examples: {len(train_data.examples)}")
print(f"Number of validation examples: {len(valid_data.examples)}")
print(f"Number of testing examples: {len(test_data.examples)}")



german.build_vocab(train_data,max_size = 10000,min_freq = 5)
english.build_vocab(train_data,max_size = 10000,min_freq = 5)

print(f"Unique tokens in source (de) vocabulary: {len(german.vocab)}")
print(f"Unique tokens in target (en) vocabulary: {len(english.vocab)}")


BATCH_SIZE = 64

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE, sort_key = lambda x:len(x.src),sort_within_batch = True,
    device = device)

for i in train_iterator:
    srcc = (i.src)
    targ = i.trg
    break;

## torch.tril(torch.ones((trg_len, trg_len)))


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(input_size,embedding_size)
        self.rnn = nn.GRU(embedding_size,hidden_size,num_layers)
        
    def forward(self,x):
        x = self.embed(x)
        output,hs =  self.rnn(x)
        
        return hs


class Decoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, num_layers):
        super(Decoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(output_size,embedding_size)
        self.rnn = nn.GRU(embedding_size,hidden_size,num_layers)
        self.fc = nn.Linear(hidden_size,output_size)
        
    def forward(self,x,hidden):
        
        x = x.unsqueeze(0)
        
        embed = self.embed(x)
        output,ht = self.rnn(embed,hidden)
        predictions = self.fc(output)
        out = predictions.squeeze(0)
        
        return out,ht


class Seq2Seq(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Seq2Seq, self).__init__()
        
        self.Encoder = Encoder
        self.Decoder = Decoder
        
    def forward(self,source,target):
        batch_size = target.shape[1]
        target_length = target.shape[0]
        target_vocab_size = len(english.vocab)
        
        output = torch.zeros(target_length, batch_size, target_vocab_size).to(device)
        
        hs = self.Encoder(source)
        x = target[0]
        
        for t in range(1,target_length):
            pred,ht = self.Decoder(x,hs)
            output[t] = pred
            best_guess = pred.argmax(1)
            
            x = best_guess
            ## For teacher forcing use 
            ##  x = target[t] if random.random() < teacher_force_ratio else best_guess
            
        return output




input_size = len(german.vocab)
output_size = len(english.vocab)
embedding_size = 100
hidden_size = 64
num_layers = 1

###
enc = Encoder(input_size ,embedding_size,hidden_size,num_layers)
dec = Decoder(output_size ,embedding_size,hidden_size,num_layers)
model = Seq2Seq(enc,dec).to(device)



num_epochs = 100
learning_rate = 0.001
batch_size = 64

TRG_PAD_IDX = english.vocab.stoi[english.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)
optimizer = optim.Adam(model.parameters())




### Train
def train(model,iterator,clip):
    
    model.train()
    epoch_loss = 0
    
    for batch_idx, batch in enumerate(iterator):
        
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)
        
        optimizer.zero_grad()
        
        out = model(inp_data, target)
        output = out[1:].view(-1, out.shape[-1])
        tar = target[1:].view(-1)
        loss = criterion(output ,tar )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss/len(iterator)


## Evaluate

def evaluate(model,iterator):
    
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(iterator):
            
            inp_data = batch.src.to(device)
            target = batch.trg.to(device)
            out = model(inp_data, target)
            output = out[1:].view(-1, out.shape[-1])
            tar = target[1:].view(-1)
        
            loss = criterion(output ,tar )
            epoch_loss += loss.item()
    return epoch_loss/len(iterator)

 n_epoch = 3
CLIP = 1
for epoch in range(n_epoch):
    t0 = time.time()
    print("Epoch: {}/{}.. ".format(epoch+1, n_epoch))
    
    train_loss = train(model,train_iterator,CLIP)
    valid_loss = evaluate(model,valid_iterator)
    
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f}')
    print('{} minutes'.format(np.round(int(time.time() - t0)/60),4))































class tokenize:
    
    def __init__(self,texts, max_features ,max_len):
        
        self.texts = texts
        self.max_features = max_features
        self.max_len = max_len
        self.token2id  = self.buildvocab()
        self.tokens = self.tokenize()

    def buildvocab(self):
        counter = Counter()
        '''

            UNK - "unknown token" - is used to replace the rare words that did not fit in your vocabulary. 
            So your sentence My name is guotong1988 will be translated into My name is _unk_.


            PAD - your GPU (or CPU at worst) processes your training data in batches and all the sequences in your batch should have the same length. 
            If the max length of your sequence is 8, your sentence .
            My name is guotong1988 will be padded from either side to fit this length: My name is guotong1988 _pad_ _pad_ _pad_ _pad_
            ---------------------------------------------------------------------------------------------------------------------------------------
            Example :: texts = ['i love barcelona' , 'barcelona is in spain','spain is in europe' , 'europe spain barcelona']
        '''

        vocab = {'<PAD>':0 , '<UNK>':1 }
        for text in self.texts:
            counter.update(text.split())

        for idx,(token , count) in enumerate(counter.most_common(self.max_features)):
            vocab.update({token:idx+2})

        ## Build tokenizer dictionary
        token2id = {k:v for v,k in enumerate(vocab.keys())}
        id2token = {v:k for v,k in enumerate(vocab.keys())}

        return token2id 
    
    def tokenize(self):
        
        return  [[(self.token2id.get(token,1)) for token in text.split()[:self.max_len]] for text in self.texts ]
    
    def padded_Sequence(self):
    
     
        padded_seqs = torch.zeros(len(self.tokens),self.max_len)
        
        for idx,seq in enumerate(self.tokens):
            padded_seqs[idx][:len(self.tokens[idx])] = torch.LongTensor(self.tokens[idx])
        
        return padded_seqs
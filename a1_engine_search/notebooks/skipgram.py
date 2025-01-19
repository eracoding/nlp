import time
import os
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import nltk
nltk.download('brown')
from nltk.corpus import brown


torch.manual_seed(42)

corpus_news = brown.sents(categories=['news'])
pprint(f"Len of sentences in news categories: {len(corpus_news)}")
corpus = [[word.lower() for word in sent] for sent in corpus_news]

# filter_chars = ['.', ',', '!', '?', ':', ';', '\n', ' ']
# freq = nltk.FreqDist([word.lower() for sent in corpus for word in sent if word not in filter_chars])
# freq.keys()

flatten = lambda l: [item for sublist in l for item in sublist]
vocabs = list(set(flatten(corpus)))

word2index = {w: i for i,w in enumerate(vocabs)}

vocabs.append("<UNK>")

word2index['<UNK>'] = len(vocabs) - 1

index2word = {v:k for k,v in word2index.items()}


def random_batch(batch_size, corpus, ws = 1):
    skip_grams = []
    for sent in corpus:
        for i in range(ws,len(sent)-ws):
            target = word2index[sent[i]]
            for j in range(ws):
                context = [word2index[sent[i -j -1]], word2index[sent[i +j +1]]]
                for w in context:
                    skip_grams.append([target, w])
    
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(skip_grams)), batch_size, replace=False) #randomly pick without replacement
        
    for i in random_index:
        random_inputs.append([skip_grams[i][0]])  # target, e.g., 2
        random_labels.append([skip_grams[i][1]])  # context word, e.g., 3
            
    return np.array(random_inputs), np.array(random_labels)



class Skipgram(nn.Module):
    
    def __init__(self, voc_size, emb_size):
        super(Skipgram, self).__init__()
        self.embedding_center  = nn.Embedding(voc_size, emb_size)
        self.embedding_outside = nn.Embedding(voc_size, emb_size)
    
    def forward(self, center, outside, all_vocabs):
        center_embedding     = self.embedding_center(center)  #(batch_size, 1, emb_size)
        outside_embedding    = self.embedding_center(outside) #(batch_size, 1, emb_size)
        all_vocabs_embedding = self.embedding_center(all_vocabs) #(batch_size, voc_size, emb_size)
        
        top_term = torch.exp(outside_embedding.bmm(center_embedding.transpose(1, 2)).squeeze(2))
        #batch_size, 1, emb_size) @ (batch_size, emb_size, 1) = (batch_size, 1, 1) = (batch_size, 1) 

        lower_term = all_vocabs_embedding.bmm(center_embedding.transpose(1, 2)).squeeze(2)
        #batch_size, voc_size, emb_size) @ (batch_size, emb_size, 1) = (batch_size, voc_size, 1) = (batch_size, voc_size) 
        
        lower_term_sum = torch.sum(torch.exp(lower_term), 1)  #(batch_size, 1)
        
        loss = -torch.mean(torch.log(top_term / lower_term_sum))  #scalar
        
        return loss


def prepare_sequence(seq, word2index):
    idxs = list(map(lambda w: word2index[w] if word2index.get(w) is not None else word2index["<UNK>"], seq))
    return torch.LongTensor(idxs)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

voc_size   = len(vocabs)
print('voc_size :', voc_size)

batch_size      = 2
embedding_size  = 2

voc_size   = len(vocabs)

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

all_vocabs = prepare_sequence(list(vocabs), word2index).expand(batch_size, voc_size).to(device)

model           = Skipgram(voc_size, embedding_size).to(device)
optimizer       = torch.optim.Adam(model.parameters(), lr=0.001)


save_path = f'./models/{model.__class__.__name__}'
os.makedirs(save_path, exist_ok=True)

num_epochs = 5000
best_loss = float('inf')
before = time.time()
for epoch in range(num_epochs):
    
    start = time.time()
    
    input_batch, label_batch = random_batch(batch_size, corpus)
    input_tensor = torch.LongTensor(input_batch).to(device)
    label_tensor = torch.LongTensor(label_batch).to(device)
    
    optimizer.zero_grad()

    loss = model(input_tensor, label_tensor, all_vocabs)
    
    loss.backward()
    
    optimizer.step()

    end = time.time()

    epoch_mins, epoch_secs = epoch_time(start, end)
    
    # torch.save(model.state_dict(), save_path)
    if (epoch + 1) % 100 == 0:  # Save every 100 epochs
        checkpoint_path = f"{save_path}/epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        # print(f"Model saved at {checkpoint_path} | Epoch: {epoch+1} | Loss: {loss:.6f}")

    elif loss < best_loss:
        best_loss = loss
        best_checkpoint_path = f"{save_path}/best.pth"
        torch.save(model.state_dict(), best_checkpoint_path)
        # print(f"New best model saved at {best_checkpoint_path} | Epoch: {epoch+1} | Loss: {loss:.6f}")
    
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch+1:6.0f} | Loss: {loss:2.6f}")

time_elapsed = time.time() - before
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
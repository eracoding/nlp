# glove.py
import math
import time
import os
import random
from pprint import pprint
from collections import Counter
from tqdm import tqdm
from itertools import combinations_with_replacement

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
# too much data when calculating co_occurence -> will reduce the data # 85981941 combinations
# corpus = corpus[:100]

# filter_chars = ['.', ',', '!', '?', ':', ';', '\n', ' ']
# freq = nltk.FreqDist([word.lower() for sent in corpus for word in sent if word not in filter_chars])
# freq.keys()

flatten = lambda l: [item for sublist in l for item in sublist]
vocabs = list(set(flatten(corpus)))

word2index = {w: i for i,w in enumerate(vocabs)}

vocabs.append("<UNK>")

word2index['<UNK>'] = len(vocabs) - 1

index2word = {v:k for k,v in word2index.items()}


X_i = Counter(flatten(corpus))

window_size = 1
skip_grams = []
for sent in corpus:
    for i in range(window_size,len(sent)-window_size):
        target = sent[i]
        for j in range(window_size):
            context = [sent[i -j -1], sent[i +j +1]]
            for w in context:
                skip_grams.append((target, w))

X_ik_skipgram = Counter(skip_grams)

def weighting(w_i, w_j, X_ik):
    try:
        x_ij = X_ik[(w_i, w_j)]
    except:
        x_ij = 1

    x_max = 100
    alpha = 0.75

    if x_ij < x_max:
        result = (x_ij / x_max) ** alpha
    else:
        result = 1

    return result



X_ik = {}
weighting_dic = {}

vocab_size = len(vocabs)
print(f"Vocabulary size: {vocab_size}")

# combs = list(combinations_with_replacement(vocabs, 2))

# print(len(combs))

for bigram in combinations_with_replacement(vocabs, 2):
    if X_ik_skipgram.get(bigram) is not None:
        co_occur = X_ik_skipgram[bigram]
        X_ik[bigram] = co_occur + 1
        X_ik[(bigram[1], bigram[0])] = co_occur + 1
    else:
        pass

    weighting_dic[bigram] = weighting(bigram[0], bigram[1], X_ik)
    weighting_dic[(bigram[1], bigram[0])] = weighting(bigram[1], bigram[0], X_ik)

# print(f"{X_ik=}")
# print(f"{weighting_dic=}")


def random_batch(batch_size, word_sequence, skip_grams, X_ik, weighting_dic):
    
    #convert to id since our skip_grams is word, not yet id
    skip_grams_id = [(word2index[skip_gram[0]], word2index[skip_gram[1]]) for skip_gram in skip_grams]
    
    random_inputs = []
    random_labels = []
    random_coocs  = []
    random_weightings = []
    random_index = np.random.choice(range(len(skip_grams_id)), batch_size, replace=False) #randomly pick without replacement
        
    for i in random_index:
        random_inputs.append([skip_grams_id[i][0]])  # target, e.g., 2
        random_labels.append([skip_grams_id[i][1]])  # context word, e.g., 3
        
        #get cooc
        pair = skip_grams[i]
        try:
            cooc = X_ik[pair]
        except:
            cooc = 1
        random_coocs.append([math.log(cooc)])
        
        #get weighting
        weighting = weighting_dic[pair]
        random_weightings.append([weighting])
                    
    return np.array(random_inputs), np.array(random_labels), np.array(random_coocs), np.array(random_weightings)


# batch_size = 2 # mini-batch size
# input_batch, target_batch, cooc_batch, weighting_batch = random_batch(batch_size, corpus, skip_grams, X_ik, weighting_dic)

# print("Input: ", input_batch)
# print("Target: ", target_batch)
# print("Cooc: ", cooc_batch)
# print("Weighting: ", weighting_batch)

class Glove(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(Glove, self).__init__()

        self.embedding_v = nn.Embedding(vocab_size, emb_size)
        self.embedding_u = nn.Embedding(vocab_size, emb_size)

        self.v_bias = nn.Embedding(vocab_size, 1)
        self.u_bias = nn.Embedding(vocab_size, 1)

    def forward(self, center_words, target_words, coocs, weighting):
        center_embeds = self.embedding_v(center_words) # [batch_size, 1, emb_size]
        target_embeds = self.embedding_u(target_words) # [batch_size, 1, emb_size]

        center_bias = self.v_bias(center_words).squeeze(1)
        target_bias = self.u_bias(target_words).squeeze(1)

        inner_product = target_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2)

        loss = weighting*torch.pow(inner_product +center_bias + target_bias - coocs, 2)

        return torch.sum(loss)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

device = torch.device("cuda:0")

batch_size     = 10
embedding_size = 2

model          = Glove(vocab_size, embedding_size).to(device)
criterion      = nn.CrossEntropyLoss()
optimizer      = torch.optim.Adam(model.parameters(), lr=0.001)

save_path = f'./models/{model.__class__.__name__}'
os.makedirs(save_path, exist_ok=True)

num_epochs = 5000
best_loss = float('inf')
before = time.time()
for epoch in range(num_epochs):
    
    start = time.time()
    
    input_batch, target_batch, cooc_batch, weighting_batch = random_batch(batch_size, corpus, skip_grams, X_ik, weighting_dic)
    input_batch  = torch.LongTensor(input_batch).to(device)         #[batch_size, 1]
    target_batch = torch.LongTensor(target_batch).to(device)        #[batch_size, 1]
    cooc_batch   = torch.FloatTensor(cooc_batch).to(device)         #[batch_size, 1]
    weighting_batch = torch.FloatTensor(weighting_batch).to(device) #[batch_size, 1]
    
    optimizer.zero_grad()
    loss = model(input_batch, target_batch, cooc_batch, weighting_batch)
    
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

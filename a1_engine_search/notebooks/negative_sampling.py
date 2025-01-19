# negative_sampling.py
import time
import os
import random
from pprint import pprint
from collections import Counter

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


def random_batch(batch_size, corpus, ws = 2):
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


# negative sampling
word_counts = Counter(flatten(corpus))

total_num_words = sum([v for v in word_counts.values()])
print(total_num_words)

z = 0.001
unigram_list = []

for v in vocabs:
    uw = word_counts[v] / total_num_words
    uw_alpha = int((uw ** 0.75) / z)
    unigram_list.extend([v] * uw_alpha)

# print(Counter(unigram_list))

def prepare_sequence(seq, word2index):
    idxs = list(map(lambda w: word2index[w] if word2index.get(w) is not None else word2index["<UNK>"], seq))
    return torch.LongTensor(idxs)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def negative_sampling(targets, unigram_table, k):
    batch_size = targets.size(0)
    neg_samples = []
    for i in range(batch_size):
        nsample = []
        target_index = targets[i].item()
        while len(nsample) < k: # num of sampling
            neg = random.choice(unigram_table)
            if word2index[neg] == target_index:
                continue
            nsample.append(neg)
        neg_samples.append(prepare_sequence(nsample, word2index).view(1, -1))
    return torch.cat(neg_samples)



class NegativeSampling(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(NegativeSampling, self).__init__()

        self.embedding_u = nn.Embedding(vocab_size, emb_size)
        self.embedding_v = nn.Embedding(vocab_size, emb_size)
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, center_words, target_words, negative_words):
        center_embs    = self.embedding_v(center_words)
        target_embs     = self.embedding_u(target_words)
        neg_embs       = self.embedding_u(negative_words)

        positive_score = target_embs.bmm(center_embs.transpose(1, 2)).squeeze(2)
        negative_score = -neg_embs.bmm(center_embs.transpose(1, 2))

        loss           = self.logsigmoid(positive_score) + torch.sum(self.logsigmoid(negative_score), 1)

        return -torch.mean(loss)

    def prediction(self, inputs):
        embeds = self.embedding_v(inputs)

        return embeds

voc_size   = len(vocabs)
print('voc_size :', voc_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
k              = 5
batch_size     = 2
embedding_size = 2
num_neg        = 10

model = NegativeSampling(voc_size, embedding_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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

    neg_samples = negative_sampling(label_tensor, unigram_list, k).to(device)
    loss = model(input_tensor, label_tensor, neg_samples)
    
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

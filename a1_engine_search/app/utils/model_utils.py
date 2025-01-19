import torch
import numpy as np
import random

def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embed(model, word, word2index):
    index = word2index.get(word, word2index['<UNK>'])
    word = torch.LongTensor([index])

    if hasattr(model, 'embedding_center'):
        embed = (model.embedding_center(word) + model.embedding_outside(word)) / 2
    else:
        embed = (model.embedding_v(word) + model.embedding_u(word)) / 2

    return np.array(embed[0].detach().numpy())

def search_similarity_top_k(model, words, word2index, vocab_embeddings, top_k=10):

    word1, word2, word3 = words
    emb_a = get_embed(model, word1, word2index)
    emb_b = get_embed(model, word2, word2index)
    emb_c = get_embed(model, word3, word2index)
    
    vector = emb_b - emb_a + emb_c

    similarity_list = [
        (vocab, float(cos_sim(vector, vocab_emb)))
        for vocab, vocab_emb in vocab_embeddings.items()
        if vocab not in [word1, word2, word3]
    ]

    top_k_results = sorted(random.sample(similarity_list, top_k), key=lambda x: x[1], reverse=True)
    # top_k_words = [word for word, _ in top_k_results]        
    
    return top_k_results

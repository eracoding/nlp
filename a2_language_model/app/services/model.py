from abc import ABC, abstractmethod
from typing import Any

import os
import random
import torch
import torchtext
import nltk
nltk.download('brown')
from nltk.corpus import brown

from app.services.all_models import Skipgram, NegativeSampling, Glove, LSTMLanguageModel
from app.utils.model_utils import get_embed, search_similarity_top_k, generate_lstm


class BaseMLModel(ABC):
    @abstractmethod
    def predict(self, req: Any) -> Any:
        raise NotImplementedError
    
    def __str__(self) -> str:
        return super().__str__()


class MLModel(BaseMLModel):
    """ML Model class"""

    def __init__(self, model_dir: list) -> None:
        self.generate_corpus(model_dir[1])

        embedding_size = 2

        skipgram = Skipgram(self.vocab_size, embedding_size)
        neg_sample = NegativeSampling(self.vocab_size, embedding_size)
        glove = Glove(self.vocab_size, embedding_size)
        self.lstm = LSTMLanguageModel(len(self.vocab_lstm), 1024, 1024, 2, 0.65)

        self.model_index = {"skipgram": 3, "glove": 0, "negative_sampling": 2, 'lstm': 1}
        self.all_models = [glove, glove, neg_sample, skipgram]


        for i, model_name in enumerate(sorted(os.listdir(model_dir[0]))):
            if '.pth' in model_name:
                model_path = os.path.join(model_dir[0], model_name)
                state_dict = torch.load(model_path, map_location=torch.device('cpu'))
                self.all_models[i].load_state_dict(state_dict)
                self.all_models[i].eval()
            elif '.pt' in model_name:
                model_path = os.path.join(model_dir[0], model_name)
                state_dict = torch.load(model_path, map_location=torch.device('cpu'))
                self.lstm.load_state_dict(state_dict)
                self.lstm.eval()

        self.vocab_embeddings = [{vocab: get_embed(model, vocab, self.word2index) for vocab in self.vocab} for model in self.all_models]

        self.prev_input, self.prev_output = None, None
    
    
    def generate_corpus(self, lstm_vocab_path: str):
        corpus_token = brown.sents(categories="news")
        self.corpus = [[word.lower() for word in sent] for sent in corpus_token]

        flatten = lambda l: [word for sent in l for word in sent]
        self.vocab = list(set(flatten(self.corpus)))

        self.word2index = {k:v for k, v in enumerate(self.vocab)}
        self.vocab.append("<UNK>")
        self.vocab_size = len(self.vocab)
        self.word2index["<UNK>"] = self.vocab_size - 1

        self.vocab_lstm = torch.load(lstm_vocab_path)
        self.tokenizer_lstm = torchtext.data.utils.get_tokenizer('basic_english')
        self.max_sex_len_lstm = 100
        self.temperature_lstm = 1.9

        self.device = torch.device('cpu')


    def predict(self, model_name: str, input_text: str) -> float:
        if self.prev_input == input_text:
            return self.prev_output
        index = self.model_index[model_name]
        model = self.all_models[index]

        if model_name == 'lstm':
            outputs = generate_lstm(input_text, self.max_sex_len_lstm, self.temperature_lstm, self.lstm, self.tokenizer_lstm, self.vocab_lstm, self.device)
        else:
            outputs = search_similarity_top_k(model, input_text, self.word2index, self.vocab_embeddings[index])
        self.prev_input, self.prev_output = input_text, outputs

        return outputs

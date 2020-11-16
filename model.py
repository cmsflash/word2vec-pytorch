import numpy as np
import scipy.stats
import torch
from torch import nn
from torch.nn import functional as f
from torchvision import models

from utils import init_embedding_, compute_wordsim_rho


class SkipGramModel(nn.Module):

    def __init__(self, vocabulary_size, dimension):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.dimension = dimension
        self.u_embedding = nn.Embedding(vocabulary_size, dimension)
        self.v_embedding = nn.Embedding(vocabulary_size, dimension)
        self.init_embeddings()

    def init_embeddings(self):
        init_embedding_(self.u_embedding.weight, self.dimension)
        nn.init.zeros_(self.v_embedding.weight)

    def forward(self, u, v, negative_v):
        u_embedding = self.u_embedding(u)
        v_embedding = self.v_embedding(v)
        unnormalized_score = (u_embedding * v_embedding).squeeze().sum(1)
        score = f.logsigmoid(unnormalized_score)
        negative_v_embedding = self.v_embedding(negative_v)
        unnormalized_negative_score = (
            negative_v_embedding @ u_embedding.unsqueeze(2)
        ).squeeze()
        negative_score = f.logsigmoid(-unnormalized_negative_score)
        return -(torch.mean(score) + torch.mean(negative_score)) / 2

    def get_wordsim_rho(self, wordsim_tuples, id_from_word, word_from_id):
        embedding = self.u_embedding.weight.cpu().detach().numpy()
        rho = compute_wordsim_rho(embedding, wordsim_tuples, id_from_word)
        return rho

    def get_embedding(self, id_from_word, word_from_id):
        embedding = self.u_embedding.weight
        return embedding

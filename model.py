import numpy as np
import scipy.stats
import torch
from torch import nn
from torch.nn import functional as f
from torchvision import models

from utils import init_embedding_, compute_wordsim_rho


# def render(char):
    # return torch.rand([char.size(0), 3, 256, 256]).cuda()


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


class NanoNet(nn.Module):

    def __init__(self, dimension):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.fc = nn.Linear(32, dimension)

        nn.init.kaiming_uniform_(self.conv1.weight)
        nn.init.kaiming_uniform_(self.conv2.weight)
        nn.init.kaiming_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # original: x = f.adaptive_avg_pool2d([1, 1]).squeeze()
        x = f.adaptive_avg_pool2d(x, output_size=1).squeeze()
        x = self.fc(x)
        return x


class GlyphNet(nn.Module):

    def __init__(self, dimension):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.fc = nn.Linear(32, dimension)

        nn.init.kaiming_uniform_(self.conv1.weight)
        nn.init.kaiming_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # original: x = f.adaptive_avg_pool2d([1, 1]).squeeze()
        x = f.adaptive_avg_pool2d(x, output_size=1).squeeze()
        x = self.fc(x)
        return x


class TianzigeCNN(nn.Module):

    def __init__(self, dimension):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 1024, 5)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(4)
        self.conv2 = nn.Conv2d(1024, 256, 1, groups=8)
        self.conv3 = nn.Conv2d(256, dimension, 2, groups=16)

        nn.init.kaiming_uniform_(self.conv1.weight)
        nn.init.kaiming_uniform_(self.conv2.weight)
        nn.init.kaiming_uniform_(self.conv3.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = f.adaptive_avg_pool2d(x, output_size=1).squeeze()
        return x


class VisualModel(nn.Module):

    def __init__(self, vocabulary_size, dimension, renderer, pretrained=False):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.dimension = dimension
        self.renderer = renderer
        self.backbone = nn.Sequential(
            models.resnet18(pretrained=True),
            nn.Linear(1000, dimension),
        )
        self.u_fc = nn.Linear(dimension, dimension)
        self.v_fc = nn.Linear(dimension, dimension)

        nn.init.xavier_uniform_(self.u_fc.weight)
        nn.init.xavier_uniform_(self.v_fc.weight)

    # u, v, negative_v are arrays of word ids
    def forward(self, u, v, negative_v):
        rendered_u = self.renderer(u).repeat(1, 3, 1, 1)
        rendered_v = self.renderer(v).repeat(1, 3, 1, 1)
        rendered_negative_v = self.renderer(negative_v)
        u_embedding = self.u_fc(self.backbone(rendered_u))
        v_embedding = self.v_fc(self.backbone(rendered_v))
        unnormalized_score = (u_embedding * v_embedding).squeeze().sum(1)
        score = f.logsigmoid(unnormalized_score)
        negative_v_embedding = torch.stack([
            self.v_fc(self.backbone(rendered_negative_v[:, i:i + 1, :, :].repeat(1, 3, 1, 1))) for i in range(0, rendered_negative_v.size(1), 3)
        ], dim=1)
        unnormalized_negative_score = (
            negative_v_embedding @ u_embedding.unsqueeze(dim=2)
        )
        negative_score = f.logsigmoid(-unnormalized_negative_score)
        loss = -(torch.mean(score) + torch.mean(negative_score)) / 2

        if torch.isnan(loss):
            pass
            #import pdb; pdb.set_trace()
        return loss

    def get_wordsim_rho(self, wordsim_tuples, id_from_word, word_from_id):
        embedding = self.get_embedding(id_from_word, word_from_id).detach().cpu().numpy()
        rho = compute_wordsim_rho(embedding, wordsim_tuples, id_from_word)
        return rho

    def get_embedding(self, id_from_word, word_from_id):
        embedding = [None for _ in word_from_id]
        for word, id_ in id_from_word.items():
            id_tensor = torch.tensor([id_]).cuda()
            word_embedding = (
                self.u_fc(self.backbone(self.renderer(id_tensor).repeat(1, 3, 1, 1)))
                .squeeze().detach().cpu().numpy()
            )
            embedding[id_from_word[word]] = word_embedding
        embedding = torch.tensor(embedding)
        return embedding


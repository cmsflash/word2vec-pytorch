import math
from os import path as osp
from input_data import InputData
import numpy
from model import SkipGramModel
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys

from utils import dump_embedding


batch_size = 50 


class Word2Vec:

    def __init__(
            self, input_path, output_dir, wordsim_path, dimension=100,
            batch_size=batch_size, window_size=5, epoch_count=1,
            initial_lr=2.5, min_count=5
        ):
        self.data = InputData(input_path, min_count)
        self.output_dir = output_dir
        self.vocabulary_size = len(self.data.id_from_word)
        self.dimension = dimension
        self.batch_size = batch_size
        self.window_size = window_size
        self.epoch_count = epoch_count
        self.initial_lr = initial_lr
        self.model = SkipGramModel(self.vocabulary_size, self.dimension)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model = nn.DataParallel(self.model.to(self.device))
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.initial_lr)

        if wordsim_path:
            self.wordsim_verification_tuples = []
            with open(wordsim_path, 'r') as f:
                f.readline() # Abandon header
                for line in f:
                    word1, word2, actual_similarity = line.split(',')
                    self.wordsim_verification_tuples.append((word1, word2, float(actual_similarity)))
        else:
            self.wordsim_verification_tuples = None

    def train(self):
        pair_count = self.data.get_pair_count(self.window_size)
        batch_count = self.epoch_count * pair_count / self.batch_size
        best_rho = 0
        for i in tqdm(range(int(batch_count)), total=batch_count):
            pos_pairs = self.data.get_batch_pairs(
                self.batch_size, self.window_size
            )
            neg_v = self.data.get_neg_v_neg_sampling(pos_pairs, 5)
            pos_u = [pair[0] for pair in pos_pairs]
            pos_v = [pair[1] for pair in pos_pairs]

            pos_u = torch.tensor(pos_u, device=self.device)
            pos_v = torch.tensor(pos_v, device=self.device)
            neg_v = torch.tensor(neg_v, device=self.device)

            self.optimizer.zero_grad()
            loss = self.model.forward(pos_u, pos_v, neg_v)
            loss.backward()
            self.optimizer.step()

            if i % 10000 == 0:
                if self.wordsim_verification_tuples:
                    rho = self.model.module.get_wordsim_rho(
                        self.wordsim_verification_tuples, self.data.id_from_word
                    )
                else:
                    rho = 0
                print(f'Loss: {loss.item()}, lr: {self.optimizer.param_groups[0]["lr"]}, rho: {rho}')
                if rho > best_rho or math.isnan(rho):
                    dump_embedding(
                        self.model.module.get_embedding(), self.model.module.dimension,
                        self.data.word_from_id, osp.join(self.output_dir, f'{i}_{rho}.txt')
                    )
                    best_rho = rho
            if i * self.batch_size % 100000 == 0:
                lr = self.initial_lr * (1.0 - 1.0 * i / batch_count)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr


if __name__ == '__main__':
    if len(sys.argv) == 3:
        sys.argv.append(None)
    w2v = Word2Vec(input_path=sys.argv[1], output_dir=sys.argv[2], wordsim_path=sys.argv[3])
    w2v.train()


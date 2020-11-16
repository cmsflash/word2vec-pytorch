import argparse
import math
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from input_data import InputData
from model import SkipGramModel
from utils import dump_embedding


batch_size = 50

class Word2Vec:

    def __init__(
        self,
        input_path,
        output_dir,
        wordsim_path,
        dimension=100,
        batch_size=batch_size,
        window_size=5,
        epoch_count=1,
        initial_lr=1e-6,
        min_count=5,
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
                f.readline()  # Abandon header
                for line in f:
                    word1, word2, actual_similarity = line.split(',')
                    self.wordsim_verification_tuples.append(
                        (word1, word2, float(actual_similarity))
                    )
        else:
            self.wordsim_verification_tuples = None

    def train(self):
        pair_count = self.data.get_pair_count(self.window_size)
        batch_count = self.epoch_count * pair_count / self.batch_size
        best_rho = float('-inf')
        for i in tqdm(range(int(batch_count)), total=batch_count):
            self.model.train()
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
            loss = self.model(pos_u, pos_v, neg_v)
            loss.backward()
            self.optimizer.step()

            if i % 250 == 0:
                self.model.eval()
                rho = self.model.module.get_wordsim_rho(
                    self.wordsim_verification_tuples, self.data.id_from_word,
                    self.data.word_from_id
                )
                print(
                    f'Loss: {loss.item()},'
                    f' lr: {self.optimizer.param_groups[0]["lr"]},'
                    f' rho: {rho}'
                )
                dump_embedding(
                    self.model.module.get_embedding(
                        self.data.id_from_word, self.data.word_from_id
                    ),
                    self.model.module.dimension,
                    self.data.word_from_id,
                    os.path.join(self.output_dir, f'latest.txt'),
                )
                if rho > best_rho:
                    dump_embedding(
                        self.model.module.get_embedding(
                            self.data.id_from_word, self.data.word_from_id
                        ),
                        self.model.module.dimension,
                        self.data.word_from_id,
                        os.path.join(self.output_dir, f'{i}_{rho}.txt')
                    )
                    best_rho = rho

            # warm up
            if i < 10000:
                lr = self.initial_lr * i / 10000
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            elif i * self.batch_size % 100000 == 0:
                lr = self.initial_lr * (1.0 - 1.0 * i / batch_count)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default='zh.txt')
    parser.add_argument('--output_dir', default='output_embedding')
    parser.add_argument('--wordsim_file', default='chinese-wordsim-297.txt')
    args = parser.parse_args()

    shutil.rmtree(args.output_dir, ignore_errors=True)
    os.makedirs(args.output_dir)

    w2v = Word2Vec(args.input_file, args.output_dir, args.wordsim_file)
    w2v.train()

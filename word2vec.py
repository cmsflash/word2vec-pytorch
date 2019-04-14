import math
import os
from os import path as osp
from input_data import InputData
from model import SkipGramModel, VisualModel
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
import argparse
from PIL import ImageFont
import numpy as np
import shutil

from utils import dump_embedding


batch_size = 50

class Word2Vec:

    def __init__(
            self, input_path, output_dir, wordsim_path, font_file, font_size, dimension=100,
            batch_size=batch_size, window_size=5, epoch_count=1,
            initial_lr=1e-6, min_count=5,
        ):
        self.data = InputData(input_path, min_count)
        self.output_dir = output_dir
        self.vocabulary_size = len(self.data.id_from_word)
        self.dimension = dimension
        self.batch_size = batch_size
        self.window_size = window_size
        self.epoch_count = epoch_count
        self.initial_lr = initial_lr
        self.font = ImageFont.truetype(font_file, font_size)
        self.model = VisualModel(self.vocabulary_size, self.dimension, self.render)
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


    def render_character(self, word_id):
        character = self.data.word_from_id[word_id]
        mask = self.font.getmask(character)
        reshaped = np.array(mask).reshape(mask.size[::-1])
        padded = np.zeros([30, 30])
        padded[:reshaped.shape[0], :reshaped.shape[1]] = reshaped
        
        return padded
        
    def render(self, word_id_tensors):
        # word_id_tensors has size:
        #     (batch_size) when rendering u and v
        #     (batch_size, negative_size) when rendering neg_v
        # output dimension: (batch_size, negative_size or 1, font_size, font_size)
        if len(word_id_tensors.size()) < 2:
            word_id_tensors = word_id_tensors.unsqueeze(1)

        return torch.Tensor([list(map(self.render_character, word_id_list)) for word_id_list in word_id_tensors.tolist()]).cuda()

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
            loss = self.model.forward(
                pos_u,
                pos_v,
                neg_v
            )
            loss.backward()
            self.optimizer.step()

            if i % 250 == 0:
                self.model.eval()
                rho = self.model.module.get_wordsim_rho(
                    self.wordsim_verification_tuples, self.data.id_from_word,
                    self.data.word_from_id
                )
                print(f'Loss: {loss.item()}, lr: {self.optimizer.param_groups[0]["lr"]}, rho: {rho}')
                dump_embedding(
                    self.model.module.get_embedding(self.data.id_from_word, self.data.word_from_id), self.model.module.dimension,
                    self.data.word_from_id, osp.join(self.output_dir, f'latest.txt')
                )
                if rho > best_rho:
                    dump_embedding(
                        self.model.module.get_embedding(self.data.id_from_word, self.data.word_from_id), self.model.module.dimension,
                        self.data.word_from_id, osp.join(self.output_dir, f'{i}_{rho}.txt')
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
    parser.add_argument('--wordsim_file', default='chinese-297-char.txt')
    parser.add_argument('--font_file', default='fonts/NotoSansCJKsc-Regular.otf')
    #parser.add_argument('--font_file', default='fonts/Adobe Heiti Std.otf')
    parser.add_argument('--font_size', default=24)
    args = parser.parse_args()

    shutil.rmtree(args.output_dir, ignore_errors=True)
    os.makedirs(args.output_dir)

    w2v = Word2Vec(
        input_path=args.input_file,
        output_dir=args.output_dir,
        wordsim_path=args.wordsim_file,
        font_file=args.font_file,
        font_size=args.font_size,
    )

    w2v.train()


import numpy
from collections import deque


numpy.random.seed(12345)


class InputData:

    def __init__(self, path, min_count):
        self.input_path = path
        self.get_words(min_count)
        self.word_pair_catch = deque()
        self.init_sample_table()

    def get_words(self, min_count):
        self.input_file = open(self.input_path)
        self.sentence_length = 0
        self.sentence_count = 0
        frequencies = dict()
        for line in self.input_file:
            self.sentence_count += 1
            line = line.strip().split(' ')
            self.sentence_length += len(line)
            for w in line:
                try:
                    frequencies[w] += 1
                except:
                    frequencies[w] = 1
        self.id_from_word = dict()
        self.word_from_id = dict()
        wid = 0
        self.frequencies = dict()
        for w, c in frequencies.items():
            if c < min_count:
                self.sentence_length -= c
                continue
            self.id_from_word[w] = wid
            self.word_from_id[wid] = w
            self.frequencies[wid] = c
            wid += 1
        self.word_count = len(self.id_from_word)

    def init_sample_table(self):
        self.sample_table = []
        sample_table_size = 1e8
        pow_frequencies = numpy.array(list(self.frequencies.values()))**0.75
        words_pow = sum(pow_frequencies)
        ratio = pow_frequencies / words_pow
        count = numpy.round(ratio * sample_table_size)
        for wid, c in enumerate(count):
            self.sample_table += [wid] * int(c)
        self.sample_table = numpy.array(self.sample_table)

    def get_batch_pairs(self, batch_size, window_size):
        while len(self.word_pair_catch) < batch_size:
            sentence = self.input_file.readline()
            if sentence is None or sentence == '':
                self.input_file = open(self.input_path)
                sentence = self.input_file.readline()
            word_ids = []
            for word in sentence.strip().split(' '):
                try:
                    word_ids.append(self.id_from_word[word])
                except:
                    continue
            for i, u in enumerate(word_ids):
                for j, v in enumerate(
                        word_ids[max(i - window_size, 0):i + window_size]):
                    assert u < self.word_count
                    assert v < self.word_count
                    if i == j:
                        continue
                    self.word_pair_catch.append((u, v))
        batch_pairs = []
        for _ in range(batch_size):
            batch_pairs.append(self.word_pair_catch.popleft())
        return batch_pairs

    def get_neg_v_neg_sampling(self, pos_word_pair, count):
        neg_v = numpy.random.choice(
            self.sample_table, size=(len(pos_word_pair), count)).tolist()
        return neg_v

    def get_pair_count(self, window_size):
        return self.sentence_length * (2 * window_size - 1) - (
            self.sentence_count - 1) * (1 + window_size) * window_size

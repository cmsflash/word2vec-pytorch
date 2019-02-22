import numpy
from collections import deque, defaultdict
from tqdm import tqdm


SAMPLE_TABLE_SIZE = int(1e8)
numpy.random.seed(12345)


class InputData:

    def __init__(self, path, min_count):
        self.input_path = path
        self.input_file = open(self.input_path)
        self.get_words(min_count)
        self.word_pair_catch = deque()
        self.sample_table = self._create_sample_table()

    def get_words(self, min_count):
        self.word_count = 0
        self.sentence_count = 0
        frequencies = defaultdict(lambda: 0)
        total=sum(1 for line in open(self.input_path))
        for line in tqdm(self.input_file, total=total, unit='lines'):
            self.sentence_count += 1
            line = line.strip().split(' ')
            self.word_count += len(line)
            for word in line:
                frequencies[word] += 1
        self.id_from_word = dict()
        self.word_from_id = dict()
        id_ = 0
        self.frequencies = dict()
        for word, frequency in frequencies.items():
            if frequency < min_count:
                self.word_count -= frequency
            else:
                self.id_from_word[word] = id_
                self.word_from_id[id_] = word
                self.frequencies[id_] = frequency
                id_ += 1
        self.vocabulary_size = len(self.id_from_word)

    def _create_sample_table(self):
        sample_table = []
        pow_frequencies = numpy.array(list(self.frequencies.values())) ** 0.75
        words_pow = sum(pow_frequencies)
        ratios = pow_frequencies / words_pow
        count = numpy.round(ratios * SAMPLE_TABLE_SIZE)
        for id_, c in enumerate(count):
            sample_table += [id_] * int(c)
        sample_table = numpy.array(sample_table)
        return sample_table

    def get_batch_pairs(self, batch_size, window_size):
        while len(self.word_pair_catch) < batch_size:
            sentence = self.input_file.readline()
            if sentence is None or sentence == '':
                self.input_file = open(self.input_path)
                sentence = self.input_file.readline()
            ids = []
            for word in sentence.strip().split(' '):
                try:
                    ids.append(self.id_from_word[word])
                except:
                    continue
            for i, u in enumerate(ids):
                for j, v in enumerate(
                    ids[max(i - window_size, 0):i + window_size]
                ):
                    assert u < self.vocabulary_size
                    assert v < self.vocabulary_size
                    if i == j:
                        continue
                    self.word_pair_catch.append((u, v))
        batch_pairs = []
        for _ in range(batch_size):
            batch_pairs.append(self.word_pair_catch.popleft())
        return batch_pairs

    def get_neg_v_neg_sampling(self, pos_word_pair, count):
        neg_v = numpy.random.choice(
            self.sample_table, size=(len(pos_word_pair), count)
        ).tolist()
        return neg_v

    def get_pair_count(self, window_size):
        return (
            self.word_count * (2 * window_size - 1)
            - (self.sentence_count - 1) * (1 + window_size) * window_size
        )

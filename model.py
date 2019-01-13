import torch
from torch import nn
from torch.nn import functional as f


class SkipGramModel(nn.Module):

    def __init__(self, word_count, dimension):
        super().__init__()
        self.word_count = word_count
        self.dimension = dimension
        self.u_embedding = nn.Embedding(word_count, dimension)
        self.v_embedding = nn.Embedding(word_count, dimension)
        self.init_embeddings()

    def init_embeddings(self):
        range_ = 0.5 / self.dimension
        nn.init.uniform_(self.u_embedding.weight, -range_, range_)
        nn.init.zeros_(self.v_embedding.weight)

    def forward(self, pos_u, pos_v, neg_v):
        u_embedding = self.u_embedding(pos_u)
        v_embedding = self.v_embedding(pos_v)
        score = torch.mul(u_embedding, v_embedding).squeeze()
        score = torch.sum(score, dim=1)
        score = f.logsigmoid(score)
        neg_v_embedding = self.v_embedding(neg_v)
        neg_score = torch.bmm(
            neg_v_embedding, u_embedding.unsqueeze(2)
        ).squeeze()
        neg_score = f.logsigmoid(-1 * neg_score)
        return -1 * (torch.mean(score)+torch.mean(neg_score))

    def save_embedding(self, id2word, file_name):
        """Save all embedding to file.

        As this class only record word id, so the map from id to word has to be transfered from outside.

        Args:
            id2word: map from word id to word.
            file_name: file name.
        Returns:
            None.
        """
        embedding = self.u_embedding.weight.cpu().data.numpy()
        fout = open(file_name, 'w')
        fout.write('%d %d\n' % (len(id2word), self.dimension))
        for wid, w in id2word.items():
            e = embedding[wid]
            e = ' '.join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (w, e))


def test():
    model = SkipGramModel(100, 100)
    id2word = dict()
    for i in range(100):
        id2word[i] = str(i)
    model.save_embedding(id2word)


if __name__ == '__main__':
    test()

from torch import nn


def init_embedding_(weight, dimension):
    range_ = 0.5 / dimension
    nn.init.uniform_(weight, -range_, range_)


def dump_embedding(embedding, dimension, word_from_id, path):
    embedding_array = embedding.cpu().data.numpy()
    with open(path, 'w') as file_:
        print(f'{len(word_from_id)} {dimension}', file=file_)
        for id_, word in word_from_id.items():
            embedding = embedding_array[id_]
            embedding_string = ' '.join(str(x) for x in embedding)
            print(f'{word} {embedding_string}', file=file_)


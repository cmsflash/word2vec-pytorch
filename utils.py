import numpy as np
import scipy.stats
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


def get_word_embedding(word, embedding, id_from_word):
    if word in id_from_word:
        # word_embedding = np.concatenate([embedding[id_from_word[word]], embedding[id_from_word[word]]], axis=0)
        word_embedding = embedding[id_from_word[word]]
    else:
        char_embeddings = []
        for char in word:
            # TODO is there a better way to handle this?
            if char in id_from_word:
                char_embeddings.append(embedding[id_from_word[char]])
            else:
                char_embeddings.append(np.zeros_like(embedding[0]))
        char_embeddings = np.array(char_embeddings)
        # word_embedding = np.concatenate([char_embeddings.min(axis=0), char_embeddings.max(axis=0)], axis=0)
        word_embedding = char_embeddings.mean(axis=0)
    return word_embedding
            

def compute_wordsim_rho(embedding, wordsim_tuples, id_from_word):
    predicted_similarities = []
    actual_similarities = []
    for word0, word1, actual_similarity in wordsim_tuples:
        embedding0 = get_word_embedding(word0, embedding, id_from_word)
        embedding1 = get_word_embedding(word1, embedding, id_from_word)
        predicted_similarity = (
            np.dot(embedding0, embedding1)
            / (np.linalg.norm(embedding0) * np.linalg.norm(embedding1))
        )
        predicted_similarities.append(predicted_similarity)
        actual_similarities.append(actual_similarity)
    spearman_rho, _ = scipy.stats.spearmanr(
        actual_similarities, predicted_similarities
    )
    spearman_rho = spearman_rho * len(actual_similarities) / len(wordsim_tuples)
    return spearman_rho


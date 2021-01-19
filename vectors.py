import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
import networkx as nx
from tqdm import tqdm
from weat import weat_score


class Embedding:
    def __init__(self, path):
        if path is None:
            self.words, self.vectors = [], []
        else:
            self.words, self.vectors = read_embeddings(path)

    def get(self, word):
        """
        Get vector for single word
        """
        index = self.words[word]
        return self.vectors[index]

    def get_many(self, words):
        """
        Get vector for a list of words
        """
        indices = [self.words[word] for word in words]
        return self.vectors[indices]


def read_embeddings(path, limit=100000):
    words = {}
    vectors = []

    with open(path, 'r') as vec_file:
        for line_idx, line in tqdm(enumerate(vec_file), total=limit, unit_scale=True):
            line = line.rstrip().split()
            words[line[0]] = line_idx
            # vector = np.array(line[1:])
            vectors.append(line[1:])

            if line_idx >= limit:
                break

    return words, np.array(vectors).astype('float')


def save(savepath, obj):
    with open(savepath, 'wb') as savefile:
        pickle.dump(obj, savefile)


def load(loadpath):
    with open(loadpath, 'rb') as loadfile:
        return pickle.load(loadfile)


def knn_graph(embedding, word_list):
    vectors = embedding.get_many(word_list)
    adjacency_matrix = kneighbors_graph(vectors, n_neighbors=3)
    graph = nx.from_scipy_sparse_matrix(adjacency_matrix)
    mapping = dict([(x, word_list[x]) for x in range(len(word_list))])
    graph = nx.relabel.relabel_nodes(graph, mapping)
    return nx.readwrite.json_graph.node_link_data(graph)


def two_means(embedding, word_list1, word_list2):
    vec1, vec2 = embedding.get_many(word_list1), embedding.get_many(word_list2)
    vec1_mean, vec2_mean = np.mean(vec1, axis=0), np.mean(vec2, axis=0)

    bias_direction = (vec1_mean - vec2_mean) / np.linalg.norm(vec1_mean - vec2_mean)
    return vec1_mean, vec2_mean, bias_direction


def compute_weat_score(embedding, X, Y, A, B):
    X_vecs, Y_vecs, A_vecs, B_vecs = [embedding.get_many(wordlist) for wordlist in [X, Y, A, B]]
    return weat_score(X_vecs, Y_vecs, A_vecs, B_vecs)


def debias_linear_projection(embedding, bias_vec):
    debiased = embedding.vectors - embedding.vectors.dot(bias_vec.reshape(-1, 1)) * bias_vec
    return debiased


def hard_debias_get_bias_direction(embedding: Embedding, word_list1: list, word_list2: list, n_components: int = 10):
    matrix = []
    for w1, w2 in zip(word_list1, word_list2):
        center = (embedding.get(w1) + embedding.get(w2)) / 2
        matrix.append(embedding.get(w1) - center)
        matrix.append(embedding.get(w2) - center)

    matrix = np.array(matrix)
    pca = PCA(n_components=n_components)
    pca.fit(matrix)
    return pca.components_[0]


def remove_component(u, v):
    return u - v * u.dot(v) / v.dot(v)


def hard_debias(base_embedding: Embedding, debiased_embedding: Embedding, bias_vec: np.ndarray, eval_words: list):
    eval_wordset = set(eval_words)
    debiased_embedding.vectors = []
    for i, word in enumerate(base_embedding.words):
        # remove bias direction from evaluation words
        if word in eval_wordset:
            debiased_embedding.vectors.append(remove_component(base_embedding.vectors[i], bias_vec))
        else:
            debiased_embedding.vectors.append(base_embedding.vectors[i])

    debiased_embedding.vectors = np.array(debiased_embedding.vectors)


if __name__ == '__main__':
    # dirty hack to make sure the object can be unpickled in the flask app
    # noinspection PyUnresolvedReferences
    from vectors import Embedding

    emb = Embedding('data/glove.6B.50d.txt')
    save('data/glove.6B.50d.pkl', emb)

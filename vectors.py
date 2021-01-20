import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.svm import LinearSVC as SVM
import networkx as nx
from tqdm import tqdm
from weat import weat_score
from enum import Enum


class WordVector:
    def __init__(self, index, word, vector):
        self.index = index
        self.word = word
        self.vector = vector

    def __str__(self):
        return f'Index={self.index}, Word="{self.word}"'

    def __repr__(self):
        return f'WordVector({self.index}, {self.word}, {self.vector})'


# Embedding class
# ----------------------------------------------------
class Embedding:
    def __init__(self, path, limit=100000):
        # either create an empty embedding (useful for creating debiased embeddings)
        if path is None:
            self.word_vectors = []
        # or read embeddings from disk
        else:
            self.word_vectors = read_embeddings(path, limit=limit)

    def get(self, word, color=''):
        """
        Get WordVector object for single word
        """
        return self.word_vectors[word]

    def get_many(self, words):
        """
        Get list of WordVector objects for a list of words
        """
        return [self.word_vectors[word] for word in words]

    def get_vecs(self, words):
        """
        Get numpy array of vectors for a given list of words
        """
        return np.vstack([self.word_vectors[word].vector for word in words])

    def vectors(self):
        return np.vstack([wv.vector for wv in self.word_vectors.values()])


# Debiaser base class
# ----------------------------------------------------
class Debiaser:
    def __init__(self, base_embedding: Embedding, debiased_embedding: Embedding):
        self.base_emb = base_embedding
        self.debiased_emb = debiased_embedding
        self.animator = Animator()

    def debias(self, bias_direction):
        # Compute debiased embedding and create animation steps here
        pass

    def get_anim_steps(self):
        pass


class LinearDebiaser(Debiaser):
    def debias(self, bias_direction):
        self.debiased_emb.vectors = self.base_emb.vectors() - self.base_emb.vectors().dot(bias_direction.reshape(-1, 1)) * bias_direction


class Animator:
    def __init__(self):
        self.anim_steps = []
        self.projectors = {}

    def add_projector(self, projector, name='GenericProjector'):
        self.projectors[name] = projector


# Various bias direction computation methods
# ----------------------------------------------------
def bias_two_means(embedding, word_list1, word_list2):
    vec1, vec2 = embedding.get_vecs(word_list1), embedding.get_vecs(word_list2)
    vec1_mean, vec2_mean = np.mean(vec1, axis=0), np.mean(vec2, axis=0)
    bias_direction = (vec1_mean - vec2_mean) / np.linalg.norm(vec1_mean - vec2_mean)

    return bias_direction


def bias_pca(embedding, word_list):
    vecs = embedding.get_vecs(word_list)
    bias_direction = PCA().fit(vecs).components_[0]

    return bias_direction


def bias_pca_paired(embedding, pair1, pair2):
    assert len(pair1) == len(pair2)
    vec1, vec2 = embedding.get_vecs(pair1), embedding.get_vecs(pair2)
    paired_vecs = vec1 - vec2
    bias_direction = PCA().fit(paired_vecs).components_[0]

    return bias_direction


def bias_classification(embedding, seedwords1, seedwords2):
    vec1, vec2 = embedding.get_vecs(seedwords1), embedding.get_vecs(seedwords2)
    x = np.vstack([vec1, vec2])
    y = np.concatenate([[0] * vec1.shape[0], [1] * vec2.shape[0]])

    classifier = SVM().fit(x, y)
    bias_direction = classifier.coef_[0]

    return bias_direction


def get_bias_direction(embedding, seedwords1, seedwords2, subspace_method):
    if subspace_method == 'Two-means':
        bias_direction = bias_two_means(embedding, seedwords1, seedwords2)
    elif subspace_method == 'PCA':
        bias_direction = bias_pca(embedding, seedwords1)
    elif subspace_method == 'PCA-paired':
        pair1, pair2 = list(zip(*[(w.split('-')[0], w.split('-')[1]) for w in seedwords1]))
        bias_direction = bias_pca_paired(embedding, pair1, pair2)
    elif subspace_method == 'Classification':
        bias_direction = bias_classification(embedding, seedwords1, seedwords2)
    else:
        raise ValueError('Incorrect subspace method')

    return bias_direction


# IO helpers
# ----------------------------------------------------
def read_embeddings(path, limit=100000):
    word_vectors = {}

    with open(path, 'r') as vec_file:
        for idx, line in tqdm(enumerate(vec_file), total=limit, unit_scale=True):
            line = line.rstrip().split()
            word = line[0]
            vector = np.array(line[1:]).astype('float')
            word_vectors[word] = WordVector(idx, word, vector)

            if idx >= limit - 1:
                break

    return word_vectors


def save(savepath, obj):
    with open(savepath, 'wb') as savefile:
        pickle.dump(obj, savefile)


def load(loadpath):
    with open(loadpath, 'rb') as loadfile:
        return pickle.load(loadfile)


# Legacy code fragments
# -----------------------------------------------------
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
    X_vecs, Y_vecs, A_vecs, B_vecs = [embedding.get_vecs(wordlist) for wordlist in [X, Y, A, B]]
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


# If called directly from command line, create and save Embedding object
if __name__ == '__main__':
    # dirty hack to make sure the object can be unpickled in the flask app
    # noinspection PyUnresolvedReferences
    from vectors import Embedding

    emb = Embedding('data/glove.6B.50d.txt')
    save('data/glove.6B.50d.pkl', emb)

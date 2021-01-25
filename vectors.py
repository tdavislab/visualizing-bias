import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.svm import LinearSVC as SVM
import networkx as nx
from tqdm import tqdm
from weat import weat_score
import scipy


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

    def words(self):
        return [wv.word for wv in self.word_vectors.values()]

    def update_vectors(self, words, debiased_vectors):
        for i, word in enumerate(words):
            self.word_vectors[word].vector = debiased_vectors[i]


# Debiaser base class
# ----------------------------------------------------
class Debiaser:
    def __init__(self, base_embedding: Embedding, debiased_embedding: Embedding):
        self.base_emb = base_embedding
        self.debiased_emb = debiased_embedding
        self.animator = Animator()

    def debias(self, bias_direction, seedwords1, seedwords2, evalwords):
        # Compute debiased embedding and create animation steps here
        raise NotImplementedError('This method should not be called from Debiaser object.')


class LinearDebiaser(Debiaser):
    def debias(self, bias_direction, seedwords1, seedwords2, evalwords):
        debiased_vectors = self.base_emb.vectors() - self.base_emb.vectors().dot(bias_direction.reshape(-1, 1)) * bias_direction
        self.debiased_emb.update_vectors(self.base_emb.words(), debiased_vectors)

        # Create base_projector for base embedding
        self.animator.add_projector(PCA(n_components=2), name='base_projector')
        base_projector = self.animator.projectors['base_projector']
        base_projector.fit(self.base_emb, seedwords1 + seedwords2)

        # Use base projector to project base embedding of seedset and evalset
        step1 = self.animator.add_anim_step()
        step1.add_points(base_projector.project(self.base_emb, seedwords1, group=1))
        step1.add_points(base_projector.project(self.base_emb, seedwords2, group=2))
        step1.add_points(base_projector.project(self.base_emb, evalwords, group=3))

        # Use base_projector to project debiased embedding of seedset and evalset to 2-d
        step2 = self.animator.add_anim_step()
        step2.add_points(base_projector.project(self.debiased_emb, seedwords1, group=1))
        step2.add_points(base_projector.project(self.debiased_emb, seedwords2, group=2))
        step2.add_points(base_projector.project(self.debiased_emb, evalwords, group=3))

        # Create debiased_projector for debiased embeddings
        self.animator.add_projector(PCA(n_components=2), name='debiased_projector')
        debiased_projector = self.animator.projectors['debiased_projector']
        debiased_projector.fit(self.debiased_emb, seedwords1 + seedwords2)

        # Use debiased projector to project debiased embedding of seedset and evalset to 2-d
        step3 = self.animator.add_anim_step()
        step3.add_points(debiased_projector.project(self.debiased_emb, seedwords1, group=1))
        step3.add_points(debiased_projector.project(self.debiased_emb, seedwords2, group=2))
        step3.add_points(debiased_projector.project(self.debiased_emb, evalwords, group=3))


class HardDebiaser(Debiaser):
    def debias(self, bias_direction, seedwords1, seedwords2, evalwords, equalize_set):
        equalize_words = list(zip(*equalize_set))

        # Create base_projector for base embedding
        self.animator.add_projector(PCA(n_components=2), name='base_projector')
        base_projector = self.animator.projectors['base_projector']
        base_projector.fit(self.base_emb, seedwords1 + seedwords2)

        # Use base projector to project base embedding of seedset and evalset
        step1 = self.animator.add_anim_step()
        step1.add_points(base_projector.project(self.base_emb, seedwords1, group=1))
        step1.add_points(base_projector.project(self.base_emb, seedwords2, group=2))
        step1.add_points(base_projector.project(self.base_emb, evalwords, group=3))
        # step1.add_points(base_projector.project(self.base_emb, equalize_words[0], group=4))
        # step1.add_points(base_projector.project(self.base_emb, equalize_words[1], group=5))

        # Remove gender component from the evalwords
        for i, word in enumerate(self.base_emb.words()):
            # remove bias direction from dataset
            self.debiased_emb.word_vectors[word].vector = remove_component(self.base_emb.word_vectors[word].vector, bias_direction)

        step2 = self.animator.add_anim_step()
        step2.add_points(base_projector.project(self.debiased_emb, seedwords1, group=1))
        step2.add_points(base_projector.project(self.debiased_emb, seedwords2, group=2))
        step2.add_points(base_projector.project(self.debiased_emb, evalwords, group=3))
        # step2.add_points(base_projector.project(self.debiased_emb, equalize_words[0], group=4))
        # step2.add_points(base_projector.project(self.debiased_emb, equalize_words[1], group=5))

        # Equalize words in equalize_set such that they are equidistant to set defining the gender direction
        for a, b in equalize_set:
            if a in self.base_emb.words() and b in self.base_emb.words():
                y = remove_component((self.base_emb.word_vectors[a].vector + self.base_emb.word_vectors[b].vector) / 2, bias_direction)
                z = np.sqrt(1 - np.linalg.norm(y) ** 2)

                if (self.base_emb.word_vectors[a].vector - self.base_emb.word_vectors[b].vector).dot(bias_direction) < 0:
                    z = -z

                self.debiased_emb.word_vectors[a] = z * bias_direction + y
                self.debiased_emb.word_vectors[b] = -z * bias_direction + y

        step3 = self.animator.add_anim_step()
        step3.add_points(base_projector.project(self.debiased_emb, seedwords1, group=1))
        step3.add_points(base_projector.project(self.debiased_emb, seedwords2, group=2))
        step3.add_points(base_projector.project(self.debiased_emb, evalwords, group=3))
        # step3.add_points(base_projector.project(self.debiased_emb, equalize_words[0], group=4))
        # step3.add_points(base_projector.project(self.debiased_emb, equalize_words[1], group=5))

        # Create debiased_projector for debiased embeddings
        self.animator.add_projector(PCA(n_components=2), name='debiased_projector')
        debiased_projector = self.animator.projectors['debiased_projector']
        debiased_projector.fit(self.debiased_emb, seedwords1 + seedwords2)

        # Use debiased projector to project debiased embedding of seedset and evalset to 2-d
        step4 = self.animator.add_anim_step()
        step4.add_points(debiased_projector.project(self.debiased_emb, seedwords1, group=1))
        step4.add_points(debiased_projector.project(self.debiased_emb, seedwords2, group=2))
        step4.add_points(debiased_projector.project(self.debiased_emb, evalwords, group=3))
        # step4.add_points(debiased_projector.project(self.debiased_emb, equalize_words[0], group=4))
        # step4.add_points(debiased_projector.project(self.debiased_emb, equalize_words[1], group=5))


class OscarDebiaser(Debiaser):
    def debias(self, bias_direction, seedwords1, seedwords2, evalwords, orth_subspace_words):
        # Create base_projector for base embedding
        self.animator.add_projector(PCA(n_components=2), name='base_projector')
        base_projector = self.animator.projectors['base_projector']
        base_projector.fit(self.base_emb, seedwords1 + seedwords2)

        # Use base projector to project base embedding of seedset and evalset
        step1 = self.animator.add_anim_step()
        step1.add_points(base_projector.project(self.base_emb, seedwords1, group=1))
        step1.add_points(base_projector.project(self.base_emb, seedwords2, group=2))
        step1.add_points(base_projector.project(self.base_emb, evalwords, group=3))
        step1.add_points(base_projector.project(self.base_emb, orth_subspace_words, group=4))

        orth_direction = bias_pca(self.base_emb, orth_subspace_words)
        rot_matrix = self.gs_constrained(np.identity(bias_direction.shape[0]), bias_direction, orth_direction)

        for word in seedwords1 + seedwords2 + evalwords + orth_subspace_words:
            self.debiased_emb.word_vectors[word].vector = self.correction(rot_matrix, bias_direction, orth_direction,
                                                                          self.base_emb.word_vectors[word].vector)

        step2 = self.animator.add_anim_step()
        step2.add_points(base_projector.project(self.debiased_emb, seedwords1, group=1))
        step2.add_points(base_projector.project(self.debiased_emb, seedwords2, group=2))
        step2.add_points(base_projector.project(self.debiased_emb, evalwords, group=3))
        step2.add_points(base_projector.project(self.debiased_emb, orth_subspace_words, group=4))

        # Create debiased_projector for debiased embeddings
        self.animator.add_projector(PCA(n_components=2), name='debiased_projector')
        debiased_projector = self.animator.projectors['debiased_projector']
        debiased_projector.fit(self.debiased_emb, seedwords1 + seedwords2)

        # Use debiased projector to project debiased embedding of seedset and evalset to 2-d
        step3 = self.animator.add_anim_step()
        step3.add_points(debiased_projector.project(self.debiased_emb, seedwords1, group=1))
        step3.add_points(debiased_projector.project(self.debiased_emb, seedwords2, group=2))
        step3.add_points(debiased_projector.project(self.debiased_emb, evalwords, group=3))
        step3.add_points(debiased_projector.project(self.debiased_emb, orth_subspace_words, group=4))

    @staticmethod
    def correction(rotation_matrix, v1, v2, x):
        def rotation(dir1, dir2, input_vec):
            def basis(vec):
                first_component = vec[0]
                second_component = vec[1]
                v2_prime = second_component - first_component * float(np.matmul(first_component, second_component.T))
                v2_prime = v2_prime / np.linalg.norm(v2_prime)
                return v2_prime

            dir1 = np.asarray(dir1).reshape(-1)
            dir2 = np.asarray(dir2).reshape(-1)
            input_vec = np.asarray(input_vec).reshape(-1)
            v2_p = basis(np.vstack((dir1, dir2)))
            x_p = input_vec[2:len(input_vec)]
            input_vec = (np.dot(input_vec, dir1), np.dot(input_vec, v2_p))
            dir2 = (np.matmul(dir2, dir1.T), np.sqrt(1 - (np.matmul(dir2, dir1.T) ** 2)))
            dir1 = (1, 0)
            theta_x = 0.0
            theta = np.abs(np.arccos(np.dot(dir1, dir2)))
            theta_p = (np.pi / 2.0) - theta
            phi = np.arccos(np.dot(dir1, input_vec / np.linalg.norm(input_vec)))
            d = np.dot([0, 1], input_vec / np.linalg.norm(input_vec))
            if phi < theta_p and d > 0:
                theta_x = theta * (phi / theta_p)
            elif phi > theta_p and d > 0:
                theta_x = theta * ((np.pi - phi) / (np.pi - theta_p))
            elif phi >= np.pi - theta_p and d < 0:
                theta_x = theta * ((np.pi - phi) / theta_p)
            elif phi < np.pi - theta_p and d < 0:
                theta_x = theta * (phi / (np.pi - theta_p))
            rotation_sincos = np.zeros((2, 2))
            rotation_sincos[0][0] = np.cos(theta_x)
            rotation_sincos[0][1] = -np.sin(theta_x)
            rotation_sincos[1][0] = np.sin(theta_x)
            rotation_sincos[1][1] = np.cos(theta_x)
            return np.hstack((np.matmul(rotation_sincos, input_vec), x_p))

        if np.count_nonzero(x) != 0:
            return np.matmul(rotation_matrix.T, rotation(v1, v2, np.matmul(rotation_matrix, x)))
        else:
            return x

    @staticmethod
    def gs_constrained(matrix, v1, v2):
        def proj(vec, a):
            return ((np.dot(vec, a.T)) * vec) / (np.dot(vec, vec))

        v1 = np.asarray(v1).reshape(-1)
        v2 = np.asarray(v2).reshape(-1)
        u = np.zeros((np.shape(matrix)[0], np.shape(matrix)[1]))
        u[0] = v1
        u[0] = u[0] / np.linalg.norm(u[0])
        u[1] = v2 - proj(u[0], v2)
        u[1] = u[1] / np.linalg.norm(u[1])
        for i in range(0, len(matrix) - 2):
            p = 0.0
            for j in range(0, i + 2):
                p = p + proj(u[j], matrix[i])
            u[i + 2] = matrix[i] - p
            u[i + 2] = u[i + 2] / np.linalg.norm(u[i + 2])
        return u


class INLPDebiaser(Debiaser):
    def debias(self, bias_direction, seedwords1, seedwords2, evalwords, num_iters=35):
        self.animator.add_projector(PCA(n_components=2), name='base_projector')
        base_projector = self.animator.projectors['base_projector']
        base_projector.fit(self.base_emb, seedwords1 + seedwords2)

        # add step
        step1 = self.animator.add_anim_step()
        step1.add_points(base_projector.project(self.base_emb, seedwords1, group=1))
        step1.add_points(base_projector.project(self.base_emb, seedwords2, group=2))
        step1.add_points(base_projector.project(self.base_emb, evalwords, group=3))

        x_projected = self.base_emb.get_vecs(seedwords1 + seedwords2).copy()
        x_eval = self.base_emb.get_vecs(evalwords).copy()

        y = np.array([0] * len(seedwords1) + [1] * len(seedwords2))
        rowspace_projections = []

        for iter_idx in range(num_iters):
            classifier_i = SVM().fit(x_projected, y)
            weights = np.expand_dims(classifier_i.coef_[0], 0)
            p_rowspace_wi = self.get_rowspace_projection(weights)
            rowspace_projections.append(p_rowspace_wi)

            # now project data to new rowspace
            P = self.get_projection_to_intersection_of_nullspaces(rowspace_projections, x_projected.shape[1])
            x_projected = P.dot(x_projected.T).T
            x_eval = P.dot(x_eval.T).T

            self.debiased_emb.update_vectors(seedwords1 + seedwords2, x_projected)
            self.debiased_emb.update_vectors(evalwords, x_eval)

            step_iter = self.animator.add_anim_step()
            step_iter.add_points(base_projector.project(self.debiased_emb, seedwords1, group=1))
            step_iter.add_points(base_projector.project(self.debiased_emb, seedwords2, group=2))
            step_iter.add_points(base_projector.project(self.debiased_emb, evalwords, group=3))

        self.animator.add_projector(PCA(n_components=2), name='debiased_projector')
        debiased_projector = self.animator.projectors['debiased_projector']
        debiased_projector.fit(self.debiased_emb, seedwords1 + seedwords2)

        step_final = self.animator.add_anim_step()
        step_final.add_points(debiased_projector.project(self.debiased_emb, seedwords1, group=1))
        step_final.add_points(debiased_projector.project(self.debiased_emb, seedwords2, group=2))
        step_final.add_points(debiased_projector.project(self.debiased_emb, evalwords, group=3))

    @staticmethod
    def get_rowspace_projection(W):
        """
        :param W: the matrix over its nullspace to project
        :return: the projection matrix over the rowspace
        """

        if np.allclose(W, 0):
            w_basis = np.zeros_like(W.T)
        else:
            w_basis = scipy.linalg.orth(W.T)  # orthogonal basis

        P_W = w_basis.dot(w_basis.T)  # orthogonal projection on W's rowspace

        return P_W

    def get_projection_to_intersection_of_nullspaces(self, rowspace_projection_matrices, input_dim):
        """
        Given a list of rowspace projection matrices P_R(w_1), ..., P_R(w_n),
        this function calculates the projection to the intersection of all nullspasces of the matrices w_1, ..., w_n.
        uses the intersection-projection formula of Ben-Israel 2013 http://benisrael.net/BEN-ISRAEL-NOV-30-13.pdf:
        N(w1)∩ N(w2) ∩ ... ∩ N(wn) = N(P_R(w1) + P_R(w2) + ... + P_R(wn))
        :param rowspace_projection_matrices: List[np.array], a list of rowspace projections
        :param dim: input dim
        """

        I = np.eye(input_dim)
        Q = np.sum(rowspace_projection_matrices, axis=0)
        P = I - self.get_rowspace_projection(Q)

        return P


class Projector:
    def __init__(self, projector, name):
        self.projector = projector
        self.name = name

    def fit(self, embedding, words):
        self.projector.fit(embedding.get_vecs(words))

    def project(self, embedding, words, group=None):
        word_vecs_2d = []
        projection = self.projector.transform(embedding.get_vecs(words))

        for i, word in enumerate(words):
            x, y = projection[i][0], projection[i][1]
            word_vecs_2d.append(WordVec2D(word, x, y, group=group))

        return word_vecs_2d


class WordVec2D:
    def __init__(self, word, x, y, group=None, meta=None):
        self.label = word
        self.x = x
        self.y = y
        self.group = group
        self.meta = meta

    def to_dict(self):
        return {'label': self.label, 'x': self.x, 'y': self.y, 'group': self.group}

    def __repr__(self):
        return f'WordVec2D("{self.label}", {self.x}, {self.y}, group={self.group}))'


class AnimStep:
    def __init__(self):
        self.points = []

    def add_points(self, word_vecs_2d):
        self.points += word_vecs_2d


class Animator:
    def __init__(self):
        self.anim_steps = []
        self.projectors = {}

    def add_projector(self, projector, name='GenericProjector'):
        self.projectors[name] = Projector(projector, name)

    def add_anim_step(self):
        new_step = AnimStep()
        self.anim_steps.append(new_step)
        return new_step

    def convert_to_payload(self):
        payload = []
        for step in self.anim_steps:
            payload.append([point.to_dict() for point in step.points])

        return payload

    def get_bounds(self):
        vectors = []

        for step in self.anim_steps:
            for point in step.points:
                vectors.append((point.x, point.y))
        vectors = np.array(vectors)

        return {
            'xmin': vectors[:, 0].min(), 'xmax': vectors[:, 0].max(),
            'ymin': vectors[:, 1].min(), 'ymax': vectors[:, 1].max()
        }


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
        bias_direction = bias_pca_paired(embedding, seedwords1, seedwords2)
    elif subspace_method == 'Classification':
        bias_direction = bias_classification(embedding, seedwords1, seedwords2)
    else:
        raise ValueError('Incorrect subspace method')

    return bias_direction


# IO helpers
def read_embeddings(path, limit=100000):
    word_vectors = {}

    with open(path, 'r') as vec_file:
        for idx, line in tqdm(enumerate(vec_file), total=limit, unit_scale=True):
            line = line.rstrip().split()
            word = line[0]
            vector = np.array(line[1:]).astype('float')
            vector = vector / np.linalg.norm(vector)
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

import numpy as np
from tqdm import tqdm


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


class Embedding:
    def __init__(self, path):
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

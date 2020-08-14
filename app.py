from flask import Flask, render_template, request, jsonify
from vectors import *

app = Flask(__name__)

# app.embedding = Embedding('data/glove.6B.50d.txt')
app.embedding = load('data/glove.6B.50d.pkl')


@app.route('/')
def index():
    return render_template('interface.html')


@app.route('/seedwords', methods=['POST'])
def get_seedwords():
    seedwords1, seedwords2 = request.values['seedwords1'], request.values['seedwords2']
    evalwords = request.values['evalwords']
    seedwords1 = [seedword.strip() for seedword in seedwords1.split(',')]
    seedwords2 = [seedword.strip() for seedword in seedwords2.split(',')]
    evalwords = [evalword.strip() for evalword in evalwords.split(',')]
    # vectors = app.embedding.get_many(seedwords)
    vectors = dim_reduction(app.embedding, seedwords1 + seedwords2 + evalwords).tolist()
    graph = knn_graph(app.embedding, seedwords1 + seedwords2)
    vectors_splits = [len(seedwords1), len(seedwords1) + len(seedwords2)]

    _, _, bias_vec = two_means(app.embedding, seedwords1, seedwords2)
    app.embedding.vectors = debias_linear_projection(app.embedding, bias_vec)
    debiased_vecs = dim_reduction(app.embedding, seedwords1 + seedwords2 + evalwords).tolist()
    return jsonify({'vectors1': vectors[:vectors_splits[0]],
                    'vectors2': vectors[vectors_splits[0]:vectors_splits[1]],
                    'evalvecs': vectors[vectors_splits[1]:],
                    'debiased_vectors1': debiased_vecs[:vectors_splits[0]],
                    'debiased_vectors2': debiased_vecs[vectors_splits[0]:vectors_splits[1]],
                    'debiased_evalvecs': debiased_vecs[vectors_splits[1]:],
                    'words1': seedwords1, 'words2': seedwords2, 'evalwords': evalwords,
                    'graph': graph})


@app.route('/weatscore', methods=['POST'])
def get_weat_score():
    male_words = {'man', 'male', 'boy', 'brother', 'him', 'his', 'son'}
    female_words = {'woman', 'female', 'girl', 'brother', 'her', 'hers', 'daughter'}

    seedwords1, seedwords2 = request.values['seedwords1'], request.values['seedwords2']
    seedwords1 = [seedword.strip() for seedword in seedwords1.split(',')]
    seedwords2 = [seedword.strip() for seedword in seedwords2.split(',')]

    return jsonify({'weat_score': compute_weat_score(app.embedding, male_words, female_words, seedwords1, seedwords2)})

from flask import Flask, render_template, request, jsonify
from vectors import Embedding, load, dim_reduction, knn_graph

app = Flask(__name__)
app.embedding = Embedding('data/glove.6B.50d.txt')


@app.route('/')
def index():
    return render_template('interface.html')


@app.route('/seedwords', methods=['POST'])
def get_seedwords():
    seedwords1, seedwords2 = request.values['seedwords1'], request.values['seedwords2']
    seedwords1 = [seedword.strip() for seedword in seedwords1.split(',')]
    seedwords2 = [seedword.strip() for seedword in seedwords2.split(',')]
    # vectors = app.embedding.get_many(seedwords)
    vectors = dim_reduction(app.embedding, seedwords1 + seedwords2).tolist()
    graph = knn_graph(app.embedding, seedwords1 + seedwords2)
    return jsonify({'vectors1': vectors[:len(seedwords1)], 'vectors2': vectors[len(seedwords1):],
                    'words1': seedwords1, 'words2': seedwords2,
                    'graph': graph})

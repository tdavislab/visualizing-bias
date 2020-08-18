from flask import Flask, render_template, request, jsonify
from vectors import *
import utils

app = Flask(__name__)

# app.embedding = Embedding('data/glove.6B.50d.txt')
app.base_embedding = load('data/glove.6B.50d.pkl')
app.debiased_embedding = Embedding(None)
app.debiased_embedding.words = app.base_embedding.words.copy()


@app.route('/')
def index():
    return render_template('interface.html')


@app.route('/seedwords', methods=['POST'])
def get_seedwords():
    seedwords1, seedwords2, evalwords = request.values['seedwords1'], request.values['seedwords2'], \
                                        request.values['evalwords']
    seedwords1 = utils.process_seedwords(seedwords1)
    seedwords2 = utils.process_seedwords(seedwords2)
    evalwords = utils.process_seedwords(evalwords)

    # Perform debiasing
    _, _, bias_direction = two_means(app.base_embedding, seedwords1, seedwords2)
    app.debiased_embedding.vectors = debias_linear_projection(app.base_embedding, bias_direction)

    projection_method = 'PCA'

    # Dimensionality reduction for vectors before debiasing
    predebiased_projector = utils.make_projector(method=projection_method)
    predebiased_projector.fit(app.base_embedding.get_many(seedwords1 + seedwords2))

    # DR for vectors after debiasing
    postdebiased_projector = utils.make_projector(method=projection_method)
    postdebiased_projector.fit(app.base_embedding.get_many(seedwords1 + seedwords2))

    weatscore_predebiased = utils.get_weat_score(app.base_embedding, seedwords1, seedwords2)
    weatscore_postdebiased = utils.get_weat_score(app.debiased_embedding, seedwords1, seedwords2)

    return jsonify({'vectors1': predebiased_projector.transform(app.base_embedding.get_many(seedwords1)).tolist(),
                    'vectors2': predebiased_projector.transform(app.base_embedding.get_many(seedwords2)).tolist(),
                    'evalvecs': predebiased_projector.transform(app.base_embedding.get_many(evalwords)).tolist(),
                    'debiased_vectors1': postdebiased_projector.transform(app.debiased_embedding.get_many(seedwords1)).tolist(),
                    'debiased_vectors2': postdebiased_projector.transform(app.debiased_embedding.get_many(seedwords2)).tolist(),
                    'debiased_evalvecs': postdebiased_projector.transform(app.debiased_embedding.get_many(evalwords)).tolist(),
                    'words1': seedwords1, 'words2': seedwords2, 'evalwords': evalwords,
                    'weat_score_predebiased': weatscore_predebiased, 'weat_score_postdebiased': weatscore_postdebiased
                    })

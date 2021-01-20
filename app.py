from flask import Flask, render_template, request, jsonify
from vectors import *
import utils
from vectors import get_bias_direction

app = Flask(__name__)

# app.embedding = Embedding('data/glove.6B.50d.txt')
app.base_embedding = load('data/glove.6B.50d.pkl')
app.debiased_embedding = load('data/glove.6B.50d.pkl')  # Embedding(None)
# app.debiased_embedding.word_vectors = app.base_embedding.word_vectors.copy()

ALGORITHMS = {
    'Algorithm: Linear debiasing': 'Linear',
    'Algorithm: Hard debiasing': 'Hard',
    'Algorithm: OSCaR': 'OSCar',
    'Algorithm: Iterative Null Space Projection': 'INLP'
}

SUBSPACE_METHODS = {
    'Subspace method: Two means': 'Two-means',
    'Subspace method: PCA': 'PCA',
    'Subspace method: PCA-paired': 'PCA-paired',
    'Subspace method: Classification': 'Classification'
}


def reload_embeddings():
    app.base_embedding = load('data/glove.6B.50d.pkl')
    app.debiased_embedding = load('data/glove.6B.50d.pkl')  # Embedding(None)
    # app.debiased_embedding.word_vectors = app.base_embedding.word_vectors.copy()


@app.route('/')
def index():
    return render_template('interface.html')


@app.route('/seedwords', methods=['POST'])
def get_seedwords():
    reload_embeddings()
    seedwords1, seedwords2, evalwords = request.values['seedwords1'], request.values['seedwords2'], request.values['evalwords']
    method, subspace_method = request.values['algorithm'], request.values['subspace_method']

    seedwords1 = utils.process_seedwords(seedwords1)
    seedwords2 = utils.process_seedwords(seedwords2)
    evalwords = utils.process_seedwords(evalwords)

    # Perform debiasing
    if method == 'Algorithm: Linear debiasing':
        _, _, bias_direction = two_means(app.base_embedding, seedwords1, seedwords2)
        app.debiased_embedding.vectors = debias_linear_projection(app.base_embedding, bias_direction)
    elif method == 'Algorithm: Hard debiasing':
        bias_direction = hard_debias_get_bias_direction(app.base_embedding, seedwords1, seedwords2)
        hard_debias(app.base_embedding, app.debiased_embedding, bias_direction, evalwords)

    projection_method = 'PCA'

    # Dimensionality reduction for vectors before debiasing
    predebiased_projector = utils.make_projector(method=projection_method)
    predebiased_projector.fit(app.base_embedding.get_many(seedwords1 + seedwords2))

    # DR for vectors after debiasing
    postdebiased_projector = utils.make_projector(method=projection_method)
    postdebiased_projector.fit(app.debiased_embedding.get_many(seedwords1 + seedwords2))

    weatscore_predebiased = utils.get_weat_score(app.base_embedding, seedwords1, seedwords2)
    weatscore_postdebiased = utils.get_weat_score(app.debiased_embedding, seedwords1, seedwords2)

    anim_steps = [
        {'vectors1': utils.project_to_2d(predebiased_projector, app.base_embedding, seedwords1),
         'words1': seedwords1,
         'vectors2': utils.project_to_2d(predebiased_projector, app.base_embedding, seedwords2),
         'words2': seedwords2,
         'evalvecs': utils.project_to_2d(predebiased_projector, app.base_embedding, evalwords),
         'evalwords': evalwords,
         'explanation': 'Projection of embedding before debiasing.'
         },
        {'vectors1': utils.project_to_2d(predebiased_projector, app.debiased_embedding, seedwords1),
         'words1': seedwords1,
         'vectors2': utils.project_to_2d(predebiased_projector, app.debiased_embedding, seedwords2),
         'words2': seedwords2,
         'evalvecs': utils.project_to_2d(predebiased_projector, app.debiased_embedding, evalwords),
         'evalwords': evalwords,
         'explanation': 'Removing the gender direction from the embedding.'
         },
        {'vectors1': utils.project_to_2d(postdebiased_projector, app.debiased_embedding, seedwords1),
         'words1': seedwords1,
         'vectors2': utils.project_to_2d(postdebiased_projector, app.debiased_embedding, seedwords2),
         'words2': seedwords2,
         'evalvecs': utils.project_to_2d(postdebiased_projector, app.debiased_embedding, evalwords),
         'evalwords': evalwords,
         'explanation': 'Projecting the points in the new debiased embedding.'
         }
    ]

    # Payload consists of:
    # 1. Initial vector embeddings
    # 2. Debiased vector embeddings
    # 3. Each step of the debiasing process, which includes the vector projection at that step, seed/eval words and explanation

    data_payload = {'vectors1': utils.project_to_2d(predebiased_projector, app.base_embedding, seedwords1),
                    'vectors2': utils.project_to_2d(predebiased_projector, app.base_embedding, seedwords2),
                    'evalvecs': utils.project_to_2d(predebiased_projector, app.base_embedding, evalwords),
                    'debiased_vectors1': utils.project_to_2d(postdebiased_projector, app.debiased_embedding, seedwords1),
                    'debiased_vectors2': utils.project_to_2d(postdebiased_projector, app.debiased_embedding, seedwords2),
                    'debiased_evalvecs': utils.project_to_2d(postdebiased_projector, app.debiased_embedding, evalwords),
                    'anim_steps': anim_steps,
                    'words1': seedwords1, 'words2': seedwords2, 'evalwords': evalwords,
                    'weat_score_predebiased': weatscore_predebiased, 'weat_score_postdebiased': weatscore_postdebiased
                    }

    all_vectors = np.vstack([data_payload['vectors1'], data_payload['vectors2'], data_payload['evalvecs'],
                             data_payload['debiased_vectors1'], data_payload['debiased_vectors2'], data_payload['debiased_evalvecs']])
    data_payload['bounds'] = {'xmin': all_vectors[:, 0].min(), 'xmax': all_vectors[:, 0].max(),
                              'ymin': all_vectors[:, 1].min(), 'ymax': all_vectors[:, 1].max()}

    return jsonify(data_payload)


@app.route('/seedwords2', methods=['POST'])
def get_seedwords2():
    reload_embeddings()
    seedwords1, seedwords2, evalwords = request.values['seedwords1'], request.values['seedwords2'], request.values['evalwords']
    algorithm, subspace_method = ALGORITHMS[request.values['algorithm']], SUBSPACE_METHODS[request.values['subspace_method']]

    seedwords1 = utils.process_seedwords(seedwords1)
    seedwords2 = utils.process_seedwords(seedwords2)
    evalwords = utils.process_seedwords(evalwords)

    if subspace_method == 'PCA-paired':
        seedwords1, seedwords2 = list(zip(*[(w.split('-')[0], w.split('-')[1]) for w in seedwords1]))

    if subspace_method == 'PCA':
        seedwords2 = []

    # Perform debiasing according to algorithm and subspace direction method
    bias_direction = get_bias_direction(app.base_embedding, seedwords1, seedwords2, subspace_method)
    print(f'Performing debiasing={algorithm} with bias_method={subspace_method}')

    if algorithm == 'Linear':
        debiaser = LinearDebiaser(app.base_embedding, app.debiased_embedding)
        debiaser.debias(bias_direction, seedwords1, seedwords2, evalwords)

    anim_steps = debiaser.animator.convert_to_payload()
    data_payload = {'base': anim_steps[0],
                    'debiased': anim_steps[-1],
                    'anim_steps': anim_steps,
                    'bounds': debiaser.animator.get_bounds()
                    }

    return jsonify(data_payload)

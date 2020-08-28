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

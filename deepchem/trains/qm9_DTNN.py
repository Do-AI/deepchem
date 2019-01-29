import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
sys.path.append('/home/hanseok/workspace/deepchem')

import pickle
import numpy as np
import tensorflow as tf
import deepchem as dc
from deepchem.models.tensorgraph.models.graph_models import DTNNModel
np.random.seed(123)
tf.set_random_seed(123)


def load_hparams(hparam_path):
    with open(hparam_path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    tasks, datasets, transformers = dc.molnet.load_qm9(featurizer='GraphConv')
    train_dataset, valid_dataset, test_dataset = datasets

    hparams = load_hparams('/home/hanseok/deepchem_results/pretrained_models/qm9dtnn/model.pickle')

    n_embedding = hparams['n_embedding']
    n_distance = hparams['n_distance']
    batch_size = hparams['batch_size']

    model = DTNNModel(
        len(tasks),
        n_embedding=n_embedding,
        n_distance=n_distance,
        output_activation=False,
        batch_size=batch_size,
        learning_rate=0.0001,
        use_queue=False,
        mode="regression",
        model_dir='/home/hanseok/deepchem_results/qm9/model'
    )

    metric = [dc.metrics.Metric(dc.metrics.mean_absolute_error)]

    best_eval = 10000000
    es = 5
    for epoch in range(100000):
        model.fit(train_dataset, nb_epoch=1, checkpoint_interval=0)

        print("Evaluating model at epoch %d" % epoch)
        train_scores = model.evaluate(train_dataset, metric, transformers)
        print("Training MAE Score: %f" % train_scores["mean-mean_absolute_error"])
        valid_scores = model.evaluate(valid_dataset, metric, transformers)
        print("Validation MAE Score: %f" % valid_scores["mean-mean_absolute_error"])

        if valid_scores["mean_absolute_error"] < best_eval:
            best_eval = valid_scores["mean_absolute_error"]
            model.save_checkpoint()
            es = 5
        elif es > 0:
            es -= 1
        else:
            test_scores = model.evaluate(test_dataset, [metric], transformers)
            print("Test MAE Score: %f" % test_scores["mean-mean_absolute_error"])
            break

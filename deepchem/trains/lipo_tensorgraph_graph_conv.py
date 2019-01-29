import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "11"
sys.path.append('/home/hanseok/workspace/deepchem')

import pickle
import numpy as np
import tensorflow as tf
import deepchem as dc
from deepchem.models.tensorgraph.models.graph_models import GraphConvModel
np.random.seed(123)
tf.set_random_seed(123)


def load_hparams(hparam_path):
    with open(hparam_path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    tasks, datasets, transformers = dc.molnet.load_lipo(featurizer='GraphConv')
    train_dataset, valid_dataset, test_dataset = datasets

    hparams = load_hparams('/home/hanseok/deepchem_results/pretrained_models/lipogcn/model.pickle')
    n_filters = hparams['n_filters']
    n_fully_connected_nodes = hparams['n_fully_connected_nodes']
    batch_size = hparams['batch_size']
    learning_rate = hparams['learning_rate']
    seed = hparams['seed']

    model = GraphConvModel(
        len(tasks),
        graph_conv_layers=[n_filters] * 2,
        dense_layer_size=n_fully_connected_nodes,
        batch_size=batch_size,
        learning_rate=learning_rate,
        random_seed=seed,
        mode='regression',
        model_dir='/home/hanseok/deepchem_results/lipogcn/model'
    )

    metric = [dc.metrics.Metric(dc.metrics.rms_score, np.mean)]

    best_eval = 10000000
    es = 5
    for epoch in range(100000):
        model.fit(train_dataset, nb_epoch=1, checkpoint_interval=0)

        print("Evaluating model at epoch %d" % epoch)
        train_scores = model.evaluate(train_dataset, metric, transformers)
        print("Training RMSE Score: %f" % train_scores["mean-rms_score"])
        valid_scores = model.evaluate(valid_dataset, metric, transformers)
        print("Validation RMSE Score: %f" % valid_scores["mean-rms_score"])

        if valid_scores["mean-rms_score"] < best_eval:
            best_eval = valid_scores["mean-rms_score"]
            model.save_checkpoint()
            es = 5
        elif es > 0:
            es -= 1
        else:
            test_scores = model.evaluate(test_dataset, metric, transformers)
            print("Test RMSE Score: %f" % test_scores["mean-rms_score"])
            break

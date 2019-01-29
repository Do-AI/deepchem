import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "8"
sys.path.append('/home/hanseok/workspace/deepchem')

import pickle
import numpy as np
import tensorflow as tf
import deepchem as dc
from deepchem.models.tensorgraph.models.graph_models import WeaveModel
np.random.seed(123)
tf.set_random_seed(123)


def load_hparams(hparam_path):
    with open(hparam_path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    tasks, datasets, transformers = dc.molnet.load_toxcast(featurizer='Weave')
    train_dataset, valid_dataset, test_dataset = datasets

    hparams = load_hparams('/home/hanseok/deepchem_results/pretrained_models/toxcastweave/model.pickle')
    n_graph_feat = hparams['n_graph_feat']
    n_pair_feat = hparams['n_pair_feat']
    learning_rate = hparams['learning_rate']
    batch_size = hparams['batch_size']

    model = WeaveModel(
        len(tasks),
        n_graph_feat=n_graph_feat,
        n_pair_feat=n_pair_feat,
        learning_rate=learning_rate,
        batch_size=batch_size,
        mode='classification',
        model_dir='/home/hanseok/deepchem_results/toxcastweave/model'
    )

    metric = dc.metrics.Metric(
        dc.metrics.roc_auc_score, np.mean, mode="classification")

    best_eval = 0
    es = 5
    for epoch in range(100000):
        model.fit(train_dataset, nb_epoch=1, checkpoint_interval=0)

        print("Evaluating model at epoch %d" % epoch)
        train_scores = model.evaluate(train_dataset, [metric], transformers)
        print("Training ROC-AUC Score: %f" % train_scores["mean-roc_auc_score"])
        valid_scores = model.evaluate(valid_dataset, [metric], transformers)
        print("Validation ROC-AUC Score: %f" % valid_scores["mean-roc_auc_score"])

        if valid_scores["mean-roc_auc_score"] > best_eval:
            best_eval = valid_scores["mean-roc_auc_score"]
            model.save_checkpoint()
            es = 5
        elif es > 0:
            es -= 1
        else:
            test_scores = model.evaluate(test_dataset, [metric], transformers)
            print("Test ROC-AUC Score: %f" % test_scores["mean-roc_auc_score"])
            break

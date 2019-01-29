import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "10"
sys.path.append('/home/hanseok/workspace/deepchem')

import pickle
import numpy as np
import tensorflow as tf
import deepchem as dc
from deepchem.models.tensorgraph.models.graph_models import MPNNModel
np.random.seed(123)
tf.set_random_seed(123)


def load_hparams(hparam_path):
    with open(hparam_path, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    delaney_tasks, delaney_datasets, transformers = dc.molnet.load_delaney(featurizer='Weave')
    train_dataset, valid_dataset, test_dataset = delaney_datasets

    hparams = load_hparams('/home/hanseok/deepchem_results/pretrained_models/esolmpnn/model.pickle')

    T = hparams['T']
    M = hparams['M']
    batch_size = hparams['batch_size']
    learning_rate = hparams['learning_rate']

    model = MPNNModel(
        len(delaney_tasks),
        T=T,
        M=M,
        batch_size=batch_size,
        n_atom_feat=75,
        n_pair_feat=14,
        learning_rate=learning_rate,
        use_queue=False,
        mode="regression",
        model_dir='/home/hanseok/deepchem_results/esol/model'
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
            model.save()
            es = 5
        elif es > 0:
            es -= 1
        else:
            test_scores = model.evaluate(test_dataset, metric, transformers)
            print("Test RMSE Score: %f" % test_scores["mean-rms_score"])
            break

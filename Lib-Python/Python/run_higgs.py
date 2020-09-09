import argparse
import gc
import glob
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
import random
from scipy import stats
from scipy.io.arff import loadarff

from imputation import KNNImputer, SimpleImputer
from spn import SPN
from learning import LearnSPN
from utils import learncats, get_stats, normalize_data, standardize_data
from trees import Tree, RandomForest
from sktrees import SKRandomForest, tree2spn


def miss(data, n):
    data = data.copy()
    for i in range(data.shape[0]):
        var = np.random.choice(np.arange(data.shape[1]), size=n, replace=False)
        data[i, var] = np.nan
    return data

if __name__ == '__main__':
    # Hyperparameters
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--run', '-r',
        type=int,
        default=0,
    )

    parser.add_argument(
        '--n_folds', '-f',
        type=int,
        default=5,
    )

    parser.add_argument(
        '--n_estimators', '-e',
        type=int,
        default=100,
    )

    parser.add_argument(
        '--msl', '-m',
        type=int,
        default=1,
    )

    FLAGS, unparsed = parser.parse_known_args()

    print('Reading File')
    data = pd.read_csv('../data/csv/higgs.csv').values
    print('Read file with shape ', data.shape)
    # Move class variable to the last column
    idx = np.append(np.arange(1, data.shape[1]), 0)
    data = data[:, idx]
    # All features are continuous except for the class variable
    ncat = np.ones(data.shape[1])
    ncat[-1] = 2

    classcol = data.shape[1]-1

    # Define how many missing values per instance to test
    n_miss = int(0.9*data.shape[1])
    if n_miss <= 10:
        step = 1
    elif n_miss <= 20:
        step = 2
    else:
        step = 3

    np.seterr(invalid='raise')

    columns=['Run', 'Number of estimators', 'Min Samples', 'N Samples']
    for i in [0] + list(range(1, n_miss, step)):
        i = str(i)
        columns += ['Naive ' +i, 'Mean '+i, 'KNN '+i, 'SPN Avg '+i, 'SPN '+i, 'LSPN ' + i, 'LSPN Avg ' + i]
    df = pd.DataFrame(columns=columns)
    df_auc = pd.DataFrame(columns=columns)

    print('####### RUN: ', FLAGS.run)
    dir = os.path.join('missing', 'higgs', str(FLAGS.run))
    Path(dir).mkdir(parents=True, exist_ok=True)

    folds = np.zeros(data.shape[0])
    for c in np.unique(data[ :, -1]):
        nn = np.sum(data[ :, -1] == c)
        ind = np.tile(np.arange(FLAGS.n_folds), int(np.ceil(nn/FLAGS.n_folds)))[:nn]
        folds[data[:, -1] == c] = np.random.choice(ind, nn, replace=False)
    np.savetxt(os.path.join(dir, 'folds.txt'), folds, fmt='%d')

    y = None
    imp_naive = {}
    imp_mean = {}
    imp_knn = {}
    proper = {}
    spn_avg = {}
    learn_spn = {}
    learn_spn_avg = {}
    auc_imp_naive = {}
    auc_imp_mean = {}
    auc_imp_knn = {}
    auc_proper = {}
    auc_spn_avg = {}
    auc_learn_spn = {}
    auc_learn_spn_avg = {}

    samples = [1000, 3000, 10000, 30000, 100000]

    for fold in range(FLAGS.n_folds):
        train_data = data[np.where(folds!=fold)[0], :]
        test_data = data[np.where(folds==fold)[0], :]

        for n_samples in samples:
            if n_samples < train_data.shape[0]:
                train_data = train_data[np.random.choice(train_data.shape[0], n_samples, replace=False), :]
            print('####### FOLD: ', fold, ' ### N SAMPLES: ', n_samples)

            # Standardize train data only
            _, maxv, minv, mean, std = get_stats(train_data, ncat)
            train_data = standardize_data(train_data, mean, std)
            test_data = standardize_data(test_data, mean, std)

            X_train, X_test = train_data[:, :-1], test_data[:, :-1]
            y_train, y_test = train_data[:, -1], test_data[:, -1]

            if y is None:
                y = y_test.copy()
            elif n_samples == samples[0]:
                y = np.append(y, y_test)

            imputer = SimpleImputer(X_train, ncat, method='mean')
            knn_imputer = KNNImputer(X_train, ncat, n_neighbors=7, approximate=True)

            rf = RandomForest(n_estimators=FLAGS.n_estimators, ncat=ncat, min_samples_leaf=FLAGS.msl)
            rf.fit(X_train, y_train)

            spn = rf.tospn()
            spn.maxv, spn.minv = maxv, minv

            lspn = rf.tospn(learnspn=True)
            lspn.maxv, lspn.minv = maxv, minv

            print('            Inference')
            nomiss, nomiss_prob = spn.classify_avg(X_test, classcol=classcol, return_prob=True)
            l_nomiss, l_nomiss_prob = lspn.classify_avg_lspn(X_test, classcol=classcol, return_prob=True)

            if fold == 0:
                imp_naive[0 + n_samples] = nomiss.copy()
                imp_mean[0 + n_samples] = nomiss.copy()
                imp_knn[0 + n_samples] = nomiss.copy()
                spn_avg[0 + n_samples] = nomiss.copy()
                learn_spn_avg[0 + n_samples] = l_nomiss.copy()

                auc_imp_naive[0 + n_samples] = nomiss_prob.copy()
                auc_imp_mean[0 + n_samples] = nomiss_prob.copy()
                auc_imp_knn[0 + n_samples] = nomiss_prob.copy()
                auc_spn_avg[0 + n_samples] = nomiss_prob.copy()
                auc_learn_spn_avg[0 + n_samples] = l_nomiss_prob.copy()


                proper[0 + n_samples], auc_proper[0 + n_samples] = spn.classify(X_test, classcol=classcol, return_prob=True)
                learn_spn[0 + n_samples], auc_learn_spn[0 + n_samples] = lspn.classify_lspn(X_test, classcol=classcol, return_prob=True)

                print("            Running missing values")
                for i in tqdm(range(1, n_miss, step)):

                    X_test_miss = miss(X_test, i)

                    proper[i + n_samples], auc_proper[i + n_samples] = spn.classify(X_test_miss, classcol=classcol, return_prob=True)
                    spn_avg[i + n_samples], auc_spn_avg[i + n_samples] = spn.classify_avg(X_test_miss, classcol=classcol, return_prob=True)

                    learn_spn[i + n_samples], auc_learn_spn[i + n_samples] = lspn.classify_lspn(X_test_miss, classcol=classcol, return_prob=True)
                    learn_spn_avg[i + n_samples], auc_learn_spn_avg[i + n_samples] = lspn.classify_avg_lspn(X_test_miss, classcol=classcol, return_prob=True)

                    imp_naive[i + n_samples], auc_imp_naive[i + n_samples] = spn.classify_avg(X_test_miss, classcol=classcol, naive=True, return_prob=True)

                    X_test_miss_1 = imputer.transform(X_test_miss)
                    imp_mean[i + n_samples], auc_imp_mean[i + n_samples] = spn.classify_avg(X_test_miss_1, classcol=classcol, return_prob=True)

                    X_test_miss_2 = knn_imputer.transform(X_test_miss)
                    imp_knn[i + n_samples], auc_imp_knn[i + n_samples] = spn.classify_avg(X_test_miss_2, classcol=classcol, return_prob=True)

            else:
                spn_avg[0 + n_samples] = np.append(spn_avg[0 + n_samples], nomiss.copy())
                learn_spn_avg[0 + n_samples] = np.append(learn_spn_avg[0 + n_samples], l_nomiss.copy())
                imp_naive[0 + n_samples] = np.append(imp_naive[0 + n_samples], nomiss.copy())
                imp_mean[0 + n_samples] = np.append(imp_mean[ + n_samples], nomiss.copy())
                imp_knn[0 + n_samples] = np.append(imp_knn[0 + n_samples], nomiss.copy())

                auc_spn_avg[0 + n_samples] = np.concatenate([auc_spn_avg[0 + n_samples], nomiss_prob.copy()], axis=0)
                auc_learn_spn_avg[0 + n_samples] = np.concatenate([auc_learn_spn_avg[0 + n_samples], l_nomiss_prob.copy()], axis=0)
                auc_imp_naive[0 + n_samples] = np.concatenate([auc_imp_naive[0 + n_samples], nomiss_prob.copy()], axis=0)
                auc_imp_mean[0 + n_samples] = np.concatenate([auc_imp_mean[0 + n_samples], nomiss_prob.copy()], axis=0)
                auc_imp_knn[0 + n_samples] = np.concatenate([auc_imp_knn[0 + n_samples], nomiss_prob.copy()], axis=0)

                pred, prob = spn.classify(X_test, classcol=classcol, return_prob=True)
                proper[0 + n_samples] = np.append(proper[0 + n_samples], pred)
                auc_proper[0 + n_samples] = np.concatenate([auc_proper[0 + n_samples], prob], axis=0)

                pred, prob = lspn.classify_lspn(X_test, classcol=classcol, return_prob=True)
                learn_spn[0 + n_samples] = np.append(learn_spn[0 + n_samples], pred)
                auc_learn_spn[0 + n_samples] = np.concatenate([auc_learn_spn[0 + n_samples], prob], axis=0)

                print("            Running missing values")
                for i in tqdm(range(1, n_miss, step)):
                    X_test_miss = miss(X_test, i)

                    pred, prob = spn.classify(X_test_miss, classcol=classcol, return_prob=True)
                    proper[i + n_samples] = np.append(proper[i + n_samples], pred)
                    auc_proper[i + n_samples] = np.concatenate([auc_proper[i + n_samples], prob], axis=0)

                    pred, prob = lspn.classify_lspn(X_test_miss, classcol=classcol, return_prob=True)
                    learn_spn[i + n_samples] = np.append(learn_spn[i + n_samples], pred)
                    auc_learn_spn[i + n_samples] = np.concatenate([auc_learn_spn[i + n_samples], prob], axis=0)

                    pred, prob = lspn.classify_avg_lspn(X_test_miss, classcol=classcol, return_prob=True)
                    learn_spn_avg[i + n_samples] = np.append(learn_spn_avg[i + n_samples], pred)
                    auc_learn_spn_avg[i + n_samples] = np.concatenate([auc_learn_spn_avg[i + n_samples], prob], axis=0)

                    pred, prob = spn.classify_avg(X_test_miss, classcol=classcol, return_prob=True)
                    spn_avg[i + n_samples] = np.append(spn_avg[i + n_samples], pred)
                    auc_spn_avg[i + n_samples] = np.concatenate([auc_spn_avg[i + n_samples], prob], axis=0)

                    pred, prob = spn.classify_avg(X_test_miss, classcol=classcol, naive=True, return_prob=True)
                    imp_naive[i + n_samples] = np.append(imp_naive[i + n_samples], pred)
                    auc_imp_naive[i + n_samples] = np.concatenate([auc_imp_naive[i + n_samples], prob], axis=0)

                    X_test_miss_1 = imputer.transform(X_test_miss)
                    pred, prob = spn.classify_avg(X_test_miss_1, classcol=classcol, return_prob=True)
                    imp_mean[i + n_samples] = np.append(imp_mean[i + n_samples], pred)
                    auc_imp_mean[i + n_samples] = np.concatenate([auc_imp_mean[i + n_samples], prob], axis=0)

                    X_test_miss_2 = knn_imputer.transform(X_test_miss)
                    pred, prob = spn.classify_avg(X_test_miss_2, classcol=classcol, return_prob=True)
                    imp_knn[i + n_samples] = np.append(imp_knn[i + n_samples], pred)
                    auc_imp_knn[i + n_samples] = np.concatenate([auc_imp_knn[i + n_samples], prob], axis=0)

            rf.delete()
            del rf
            spn.delete()
            del spn
            lspn.delete()
            del lspn
            gc.collect()

    for n_samples in samples:
        d, d_auc = {}, {}
        d['Run'], d_auc['Run'] = FLAGS.run, FLAGS.run
        d['Number of estimators'], d_auc['Number of estimators'] = 100, 100
        d['Min Samples'], d_auc['Min Samples'] = FLAGS.msl, FLAGS.msl
        d['N Samples'], d_auc['N Samples'] = n_samples, n_samples

        enc = OneHotEncoder()
        auc_y = enc.fit_transform(y.reshape(-1, 1)).toarray()

        for i in [0] + list(range(1, n_miss, step)):
            j = str(i)
            d['Naive '+j] = np.mean(imp_naive[i + n_samples] == y)
            d['Mean '+j] = np.mean(imp_mean[i + n_samples] == y)
            d['KNN '+j] = np.mean(imp_knn[i + n_samples] == y)
            d['SPN Avg '+j] = np.mean(spn_avg[i + n_samples] == y)
            d['SPN '+j] = np.mean(proper[i + n_samples] == y)
            d['LSPN '+j] = np.mean(learn_spn[i + n_samples] == y)
            d['LSPN Avg '+j] = np.mean(learn_spn_avg[i + n_samples] == y)

            d_auc['Naive '+j] = roc_auc_score(auc_y, auc_imp_naive[i + n_samples])
            d_auc['Mean '+j] = roc_auc_score(auc_y, auc_imp_mean[i + n_samples])
            d_auc['KNN '+j] = roc_auc_score(auc_y, auc_imp_knn[i + n_samples])
            d_auc['SPN Avg '+j] = roc_auc_score(auc_y, auc_spn_avg[i + n_samples])
            d_auc['SPN '+j] = roc_auc_score(auc_y, auc_proper[i + n_samples])
            d_auc['LSPN '+j] = roc_auc_score(auc_y, auc_learn_spn[i + n_samples])
            d_auc['LSPN Avg '+j] = roc_auc_score(auc_y, auc_learn_spn_avg[i + n_samples])
        df = df.append(d, ignore_index=True)
        df_auc = df.append(d_auc, ignore_index=True)
        df.to_csv(os.path.join('missing', 'higgs_' + str(FLAGS.run) + '.csv'))
        df_auc.to_csv(os.path.join('missing', 'auc_higgs_' + str(FLAGS.run) + '.csv'))

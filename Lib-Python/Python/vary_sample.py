import argparse
import gc
import glob
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import _tree, DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn import tree
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
from utils import str2bool, learncats, get_stats, normalize_data
from trees import Tree, RandomForest
from sktrees import SKRandomForest, tree2spn


# Auxiliary functions
def get_dummies(data):
    data = data.copy()
    if isinstance(data, pd.Series):
        data = pd.factorize(data)[0]
        return data
    for col in data.columns:
        data.loc[:, col] = pd.factorize(data[col])[0]
    return data


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
        '--dataset', '-d',
        type=str,
        default='sachs',
    )

    parser.add_argument(
        '--runs', '-r',
        nargs='+',
        type=int,
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    )

    parser.add_argument(
        '--msl', '-l',
        type=int,
        default=1,
    )

    parser.add_argument(
        '--n_estimators', '-e',
        type=int,
        default=100,
    )

    parser.add_argument(
        '--ul', '-u',
        type=str2bool,
        default='false',
    )

    FLAGS, unparsed = parser.parse_known_args()

    # Read and preprocess the data
    if 'sachs' in FLAGS.dataset:
        all_data = pd.read_csv('../data/csv/sachs1000000.csv', delim_whitespace=True, comment='#')
        all_data = all_data.drop(0)
        ncat = learncats(all_data.drop_duplicates().values.astype(float), classcol=-1)
        all_data = all_data.values.astype(float)
        classcol = all_data.shape[1]-1

    # Define how many missing values per instance to test
    n_miss = int(0.9*all_data.shape[1])
    if n_miss <= 10:
        step = 1
    elif n_miss <= 20:
        step = 2
    else:
        step = 3

    np.seterr(invalid='raise')

    columns=['Run', 'Number of estimators', 'Min Samples at Leaves', 'N Samples']
    for i in [0] + list(range(1, n_miss, step)):
        i = str(i)
        columns += ['Naive ' +i, 'Mean '+i, 'KNN '+i, 'SPN Avg '+i, 'SPN '+i, 'LSPN ' + i, 'LSPN Avg ' + i]
    df = pd.DataFrame(columns=columns)
    df_auc = pd.DataFrame(columns=columns)

    for run in FLAGS.runs:
        print('####### DATASET: ', FLAGS.dataset)
        print('####### RUN: ', run)
        dir = os.path.join('missing', FLAGS.dataset, str(run))

        np.random.seed(run)  # Easy way to reproduce the folds
        test_idx = np.random.choice(range(all_data.shape[0]), size=20000, replace=False)
        data_test = all_data[test_idx, :]
        data_train = all_data[~test_idx, :]

        for n_samples in [1000, 3000, 10000, 30000, 100000]:
            print('####### N SAMPLES: ', n_samples)
            y = None
            # Normalize on train data only
            train_idx = np.random.choice(range(data_train.shape[0]), size=n_samples, replace=False)
            train_data = data_train[train_idx, :].copy()
            test_data = data_test.copy()

            train_data, maxv, minv, _, _ = get_stats(train_data, ncat)
            test_data = normalize_data(test_data, maxv, minv)

            X_train, X_test = train_data[:, :-1], test_data[:, :-1]
            y_train, y_test = train_data[:, -1], test_data[:, -1]

            if y is None:
                y = y_test.copy()
            else:
                y = np.append(y, y_test)

            imputer = SimpleImputer(X_train, ncat, method='mean')

            knn_imputer = KNNImputer(X_train, ncat, n_neighbors=7, approximate=False)

            print('            Training')
            rf = RandomForest(n_estimators=FLAGS.n_estimators, ncat=ncat, min_samples_leaf=FLAGS.msl)
            rf.fit(X_train, y_train)

            spn = rf.tospn()
            spn.maxv, spn.minv = maxv, minv

            lspn = rf.tospn(learnspn=True)
            lspn.maxv, lspn.minv = maxv, minv

            print('            Inference')
            nomiss, nomiss_prob = spn.classify_avg(X_test, classcol=classcol, return_prob=True)
            nomiss_prob = nomiss_prob.reshape(-1, int(ncat[-1]))
            l_nomiss, l_nomiss_prob = lspn.classify_avg_lspn(X_test, classcol=classcol, return_prob=True)
            l_nomiss_prob = l_nomiss_prob.reshape(-1, int(ncat[-1]))

            imp_naive = {}
            imp_mean = {}
            imp_knn = {}
            proper = {}
            spn_avg = {}
            learn_spn = {}
            learn_spn_avg = {}

            imp_naive[0] = nomiss.copy()
            imp_mean[0] = nomiss.copy()
            imp_knn[0] = nomiss.copy()
            spn_avg[0] = nomiss.copy()
            learn_spn_avg[0] = l_nomiss.copy()

            auc_imp_naive = {}
            auc_imp_mean = {}
            auc_imp_knn = {}
            auc_proper = {}
            auc_spn_avg = {}
            auc_learn_spn = {}
            auc_learn_spn_avg = {}

            auc_imp_naive[0] = nomiss_prob.copy()
            auc_imp_mean[0] = nomiss_prob.copy()
            auc_imp_knn[0] = nomiss_prob.copy()
            auc_spn_avg[0] = nomiss_prob.copy()
            auc_learn_spn_avg[0] = l_nomiss_prob.copy()


            proper[0], auc_proper[0] = spn.classify(X_test, classcol=classcol, return_prob=True)
            learn_spn[0], auc_learn_spn[0] = lspn.classify_lspn(X_test, classcol=classcol, return_prob=True)

            print("            Running missing values")
            for i in tqdm(range(1, n_miss, step)):

                X_test_miss = miss(X_test, i)

                proper[i], auc_proper[i] = spn.classify(X_test_miss, classcol=classcol, return_prob=True)
                auc_proper[i] = auc_proper[i].reshape(-1, int(ncat[-1]))
                spn_avg[i], auc_spn_avg[i] = spn.classify_avg(X_test_miss, classcol=classcol, return_prob=True)
                auc_spn_avg[i] = auc_spn_avg[i].reshape(-1, int(ncat[-1]))

                learn_spn[i], auc_learn_spn[i] = lspn.classify_lspn(X_test_miss, classcol=classcol, return_prob=True)
                auc_learn_spn[i] = auc_learn_spn[i].reshape(-1, int(ncat[-1]))
                learn_spn_avg[i], auc_learn_spn_avg[i] = lspn.classify_avg_lspn(X_test_miss, classcol=classcol, return_prob=True)
                auc_learn_spn_avg[i] = auc_learn_spn_avg[i].reshape(-1, int(ncat[-1]))

                imp_naive[i], auc_imp_naive[i] = spn.classify_avg(X_test_miss, classcol=classcol, naive=True, return_prob=True)
                auc_imp_naive[i] = auc_imp_naive[i].reshape(-1, int(ncat[-1]))

                X_test_miss_1 = imputer.transform(X_test_miss)
                imp_mean[i], auc_imp_mean[i] = spn.classify_avg(X_test_miss_1, classcol=classcol, return_prob=True)
                auc_imp_mean[i] = auc_imp_mean[i].reshape(-1, int(ncat[-1]))

                X_test_miss_2 = knn_imputer.transform(X_test_miss) # knn_imputer.transform(X_test_miss)
                imp_knn[i], auc_imp_knn[i] = spn.classify_avg(X_test_miss_2, classcol=classcol, return_prob=True)
                auc_imp_knn[i] = auc_imp_knn[i].reshape(-1, int(ncat[-1]))

            rf.delete()
            del rf
            spn.delete()
            del spn
            lspn.delete()
            del lspn
            gc.collect()

            d, d_auc = {}, {}
            d['Run'], d_auc['Run'] = run, run
            d['Number of estimators'], d_auc['Number of estimators'] = 100, 100
            d['Min Samples at Leaves'], d_auc['Min Samples at Leaves'] = FLAGS.msl, FLAGS.msl
            d['N Samples'], d_auc['N Samples'] = n_samples, n_samples

            enc = OneHotEncoder()
            auc_y = enc.fit_transform(y.reshape(-1, 1)).toarray()

            for i in [0] + list(range(1, n_miss, step)):
                j = str(i)
                d['Naive '+j] = np.mean(imp_naive[i] == y)
                d['Mean '+j] = np.mean(imp_mean[i] == y)
                d['KNN '+j] = np.mean(imp_knn[i] == y)
                d['SPN Avg '+j] = np.mean(spn_avg[i] == y)
                d['SPN '+j] = np.mean(proper[i] == y)
                d['LSPN '+j] = np.mean(learn_spn[i] == y)
                d['LSPN Avg '+j] = np.mean(learn_spn_avg[i] == y)

                d_auc['Naive '+j] = roc_auc_score(auc_y, auc_imp_naive[i], average='weighted')
                d_auc['Mean '+j] = roc_auc_score(auc_y, auc_imp_mean[i], average='weighted')
                d_auc['KNN '+j] = roc_auc_score(auc_y, auc_imp_knn[i], average='weighted')
                d_auc['SPN Avg '+j] = roc_auc_score(auc_y, auc_spn_avg[i], average='weighted')
                d_auc['SPN '+j] = roc_auc_score(auc_y, auc_proper[i], average='weighted')
                d_auc['LSPN '+j] = roc_auc_score(auc_y, auc_learn_spn[i], average='weighted')
                d_auc['LSPN Avg '+j] = roc_auc_score(auc_y, auc_learn_spn_avg[i], average='weighted')
            df = df.append(d, ignore_index=True)
            df_auc = df_auc.append(d_auc, ignore_index=True)
            df.to_csv(os.path.join('missing', FLAGS.dataset + str(FLAGS.msl) + '.csv'), index=False)
            df_auc.to_csv(os.path.join('missing', 'auc_' + FLAGS.dataset + str(FLAGS.msl) + '.csv'), index=False)

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

from knn_imputer import KNNImputer
import prep
from simple_imputer import SimpleImputer
from spn import SPN
from learning import LearnSPN
from utils import str2bool, learncats, get_stats, normalize_data, standardize_data
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
        default='wine',
    )

    parser.add_argument(
        '--runs', '-r',
        nargs='+',
        type=int,
        default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
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
        '--ul', '-u',
        type=str2bool,
        default='false',
    )

    parser.add_argument(
        '--msl', '-m',
        type=int,
        default=1,
    )

    parser.add_argument(
        '--sklearn', '-sk',
        type=str2bool,
        default='false',
    )

    parser.add_argument(
        '--surrogate', '-s',
        type=str2bool,
        default='true',
    )

    FLAGS, unparsed = parser.parse_known_args()

    # Read and preprocess the data
    if 'wine' in FLAGS.dataset:
        data_red = pd.read_csv('../data/csv/winequality_red.csv')
        data_white = pd.read_csv('../data/csv/winequality_white.csv')
        data = pd.concat([data_red, data_white]).values
        data[:, -1] = np.where(data[:, -1] <= 6, 0, 1)
        ncat = learncats(data, classcol=data.shape[1]-1)
    elif 'bank' in FLAGS.dataset:
        data = pd.read_csv('../data/csv/bank-additional-full.csv', sep=';')
        data, ncat = prep.bank(data)
    elif 'segment' in FLAGS.dataset:
        data = pd.read_csv('../data/automl/segment.csv')
        data, ncat = prep.segment(data)
    elif 'credit' in FLAGS.dataset:
        data = pd.read_csv('../data/csv/credit.csv')
        data, ncat = prep.credit(data)
    elif 'adult' in FLAGS.dataset:
        data = pd.read_csv('../data/csv/adults.csv')
        data, ncat = prep.adult(data)
    elif 'australia' in FLAGS.dataset:
        data = pd.read_csv('../data/automl/australian.csv')
        data, ncat = prep.australia(data)
    elif 'german' in FLAGS.dataset:
        data = pd.read_csv('../data/csv/german.csv', sep=' ', header=None)
        data, ncat = prep.german(data)
    elif 'car' in FLAGS.dataset:
        data = pd.read_csv('../data/automl/car.csv')
        data = get_dummies(data).values.astype(float)
        ncat = learncats(data, classcol=-1)
    elif 'vehicle' in FLAGS.dataset:
        data = pd.read_csv('../data/automl/vehicle.csv')
        data['Class'] = get_dummies(data['Class'])
        ncat = np.ones(data.shape[1])
        ncat[-1] = len(np.unique(data['Class']))
        data = data.values.astype(float)
    elif 'higgs' in FLAGS.dataset:
        data = pd.read_csv('../data/csv/higgs.csv').values
        print('Read file with shape ', data.shape)
        # Move class variable to the last column
        idx = np.append(np.arange(1, data.shape[1]), 0)
        data = data[:, idx]
        # All features are continuous except for the class variable
        ncat = np.ones(data.shape[1])
        ncat[-1] = 2
        FLAGS.ul = 'per_leaf'
    elif 'iono' in FLAGS.dataset:
        data = pd.read_csv('../data/csv/ionosphere.csv', header=None)
        data = data.drop(1, axis=1)  # ignore constant attribute
        data.iloc[:,-1] = get_dummies(data.iloc[:,-1])
        data = data.values
        ncat = learncats(data, classcol=data.shape[1]-1).astype(np.int64)
    elif 'pendigits' in FLAGS.dataset:
        data = pd.read_csv('../data/csv/pendigits.csv').values.astype(float)
        ncat = learncats(data, classcol=data.shape[1]-1)
    elif 'vowel' in FLAGS.dataset:
        data = pd.read_csv('../data/csv/vowel.csv')
        data, ncat = prep.vowel(data)
    elif 'authent' in FLAGS.dataset:
        data = pd.read_csv('../data/csv/authent.csv')
        data['Class'] = get_dummies(data['Class'])
        ncat = learncats(data.values).astype(int)
        data = data.values.astype(float)
    elif 'diabetes' in FLAGS.dataset:
        data = pd.read_csv('../data/csv/diabetes.csv')
        data['class'] = get_dummies(data['class'])
        ncat = learncats(data.values,
                         continuous_ids=[0] # Force first variable to be continuous
                         ).astype(int)
        data = data.values.astype(float)
    elif 'cmc' in FLAGS.dataset:
        data = pd.read_csv('../data/csv/cmc.csv')
        data, ncat = prep.cmc(data)

    filepath = os.path.join('train_missing', FLAGS.dataset + str(FLAGS.runs[0]) + '_train.csv')

    np.seterr(invalid='raise')

    df_out = pd.DataFrame()

    for run in FLAGS.runs:
        print('####### DATASET: ', FLAGS.dataset)
        print('####### RUN: ', run)
        dir = os.path.join('train_missing', FLAGS.dataset, str(run))

        Path(dir).mkdir(parents=True, exist_ok=True)

        np.random.seed(run)  # Easy way to reproduce the folds
        folds = np.zeros(data.shape[0])
        for c in np.unique(data[ :, -1]):
            nn = np.sum(data[ :, -1] == c)
            ind = np.tile(np.arange(FLAGS.n_folds), int(np.ceil(nn/FLAGS.n_folds)))[:nn]
            folds[data[:, -1] == c] = np.random.choice(ind, nn, replace=False)
        np.savetxt(os.path.join(dir, 'folds.txt'), folds, fmt='%d')

        for missing_perc in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            print('####### MISSING ', missing_perc)
            y = None
            for fold in range(FLAGS.n_folds):
                train_data = data[np.where(folds!=fold)[0], :]
                test_data = data[np.where(folds==fold)[0], :]

                # Standardize train data only
                _, maxv, minv, mean, std = get_stats(train_data, ncat)
                train_data = standardize_data(train_data, mean, std)
                test_data = standardize_data(test_data, mean, std)

                X_train, X_test = train_data[:, :-1].copy(), test_data[:, :-1].copy()
                y_train, y_test = train_data[:, -1].copy(), test_data[:, -1].copy()

                missing_mask = np.full(X_train.size, False)
                missing_mask[:int(missing_perc * X_train.size)] = True
                np.random.shuffle(missing_mask)
                missing_mask = missing_mask.astype(bool)
                X_train.ravel()[missing_mask] = np.nan

                print('########## MISSING ', np.sum(np.isnan(X_train)), ' instances.')

                if y is None:
                    y = y_test.copy()
                else:
                    y = np.append(y, y_test)

                imputer = SimpleImputer(ncat=ncat[:-1], method='mean').fit(X_train)

                knn_imputer = KNNImputer(ncat=ncat[:-1], n_neighbors=7).fit(X_train)

                print('            Training')
                if FLAGS.sklearn:
                    rf = SKRandomForest(n_estimators=FLAGS.n_estimators, ncat=ncat, min_samples_leaf=FLAGS.msl)
                    rf.fit(X_train, y_train)
                else:
                    rf = RandomForest(n_estimators=FLAGS.n_estimators, ncat=ncat, min_samples_leaf=FLAGS.msl, surrogate=FLAGS.surrogate)
                    rf.fit(X_train, y_train)

                print('            Converting to SPN')
                spn = rf.tospn(uniformleaves=FLAGS.ul)
                spn.maxv, spn.minv = maxv, minv

                lspn = rf.tospn(learnspn=True, uniformleaves=FLAGS.ul)
                lspn.maxv, lspn.minv = maxv, minv

                print("            Inference")
                for i in tqdm(np.arange(0, 1., 0.1)):
                    df = pd.DataFrame()
                    if i == 0:
                        nomiss, nomiss_prob = spn.classify_avg(X_test, classcol=data.shape[1]-1, return_prob=True)
                        l_nomiss, l_nomiss_prob = lspn.classify_avg_lspn(X_test, classcol=data.shape[1]-1, return_prob=True)
                        proper, auc_proper = spn.classify(X_test, classcol=data.shape[1]-1, return_prob=True)
                        learn_spn, auc_learn_spn = lspn.classify_lspn(X_test, classcol=data.shape[1]-1, return_prob=True)

                        imp_naive = nomiss
                        imp_mean = nomiss
                        imp_knn = nomiss
                        spn_avg = nomiss
                        learn_spn_avg = l_nomiss

                        auc_imp_naive = nomiss_prob
                        auc_imp_mean = nomiss_prob
                        auc_imp_knn = nomiss_prob
                        auc_spn_avg = nomiss_prob
                        auc_learn_spn_avg = l_nomiss_prob
                    else:
                        missing_mask = np.full(X_test.size, False)
                        missing_mask[:int(i * X_test.size)] = True
                        np.random.shuffle(missing_mask)
                        missing_mask = missing_mask.astype(bool)

                        X_test_miss = X_test.copy()
                        X_test_miss.ravel()[missing_mask] = np.nan

                        X_test_miss_1 = imputer.transform(X_test_miss)
                        imp_mean, auc_imp_mean = spn.classify_avg(X_test_miss_1, classcol=data.shape[1]-1, return_prob=True)

                        X_test_miss_2 = knn_imputer.transform(X_test_miss)
                        imp_knn, auc_imp_knn = spn.classify_avg(X_test_miss_2, classcol=data.shape[1]-1, return_prob=True)

                        proper, auc_proper = spn.classify(X_test_miss, classcol=data.shape[1]-1, return_prob=True)
                        spn_avg, auc_spn_avg = spn.classify_avg(X_test_miss, classcol=data.shape[1]-1, return_prob=True)

                        learn_spn, auc_learn_spn = lspn.classify_lspn(X_test_miss, classcol=data.shape[1]-1, return_prob=True)
                        learn_spn_avg, auc_learn_spn_avg = lspn.classify_avg_lspn(X_test_miss, classcol=data.shape[1]-1, return_prob=True)

                        imp_naive, auc_imp_naive = spn.classify_avg(X_test_miss, classcol=data.shape[1]-1, naive=True, return_prob=True)

                    df['True'] = y_test
                    df['Naive'] = imp_naive
                    df['Mean'] = imp_mean
                    df['KNN'] = imp_knn
                    df['SPN'] = proper
                    df['SPN Avg'] = spn_avg
                    df['LSPN'] = learn_spn
                    df['LSPN Avg'] = learn_spn_avg

                    df['p Naive'] = auc_imp_naive.tolist()
                    df['p Mean'] = auc_imp_mean.tolist()
                    df['p KNN'] = auc_imp_knn.tolist()
                    df['p SPN'] = auc_proper.tolist()
                    df['p SPN Avg'] = auc_spn_avg.tolist()
                    df['p LSPN'] = auc_learn_spn.tolist()
                    df['p LSPN Avg'] = auc_learn_spn_avg.tolist()

                    df['Missing Test'] = i
                    df['Missing Train'] = missing_perc
                    df['Run'] = run
                    df['Fold'] = fold
                    df['Number of estimators'] = 100
                    df['Min Samples'] = FLAGS.msl

                    # if file does not exist write header
                    if not os.path.isfile(filepath):
                       df.to_csv(filepath, index=False)
                    else: # else it exists so append without writing the header
                        df.to_csv(filepath, mode='a', index=False, header=False)

                if not FLAGS.sklearn:
                    rf.delete()
                    del rf
                spn.delete()
                del spn
                lspn.delete()
                del lspn
                gc.collect()

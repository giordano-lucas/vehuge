import numpy as np
import pandas as pd
from scipy.io.arff import loadarff

from utils import learncats


# Auxiliary functions
def get_dummies(data):
    data = data.copy()
    if isinstance(data, pd.Series):
        data = pd.factorize(data)[0]
        return data
    for col in data.columns:
        data.loc[:, col] = pd.factorize(data[col])[0]
    return data


# Preprocessing functions
def adult(data):
    cat_cols = ['workclass', 'education', 'education-num', 'marital-status', 'occupation',
               'relationship', 'race', 'sex', 'native-country', 'y']
    cont_cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'capital-gain',
                'capital-loss', 'hours-per-week']
    data.loc[:, cat_cols] = get_dummies(data[cat_cols])
    ncat = learncats(data.values, classcol=-1, continuous_ids=[data.columns.get_loc(c) for c in cont_cols])
    return data.values.astype(float), ncat


def australia(data):
    cat_cols = ['A1', 'A4', 'A5', 'A6', 'A8', 'A9', 'A11', 'A12', 'A15']
    cont_cols = ['A2', 'A3', 'A7', 'A10', 'A13', 'A14']
    data.loc[:, cat_cols] = get_dummies(data[cat_cols])
    ncat = learncats(data.values, classcol=-1, continuous_ids=[data.columns.get_loc(c) for c in cont_cols])
    return data.values.astype(float), ncat


def bank(data):
    cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan',
       'contact', 'month', 'day_of_week', 'poutcome', 'y']
    cont_cols = ['age', 'duration', 'campaign', 'previous', 'emp.var.rate',
                'cons.price.idx','cons.conf.idx', 'euribor3m', 'nr.employed']
    data.loc[:, cat_cols] = get_dummies(data[cat_cols])
    data.loc[:, 'pdays'] = np.where(data['pdays']==999, 0, 1)
    ncat = learncats(data.values, classcol=-1, continuous_ids=[data.columns.get_loc(c) for c in cont_cols])
    return data.values.astype(float), ncat


def credit(data):
    cat_cols = ['SEX', 'EDUCATION', 'MARRIAGE', 'default payment next month']
    cont_cols = ['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2',
       'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    data.loc[:, cat_cols] = get_dummies(data[cat_cols])
    ncat = learncats(data.values, classcol=-1, continuous_ids=[data.columns.get_loc(c) for c in cont_cols])
    return data.values.astype(float), ncat


def segment(data):
    data = data.drop(columns=['region.centroid.col', 'region.pixel.count'])
    cat_cols = ['short.line.density.5', 'short.line.density.2', 'class']
    cont_cols = ['region.centroid.row', 'vedge.mean', 'vegde.sd', 'hedge.mean', 'hedge.sd',
                 'intensity.mean', 'rawred.mean', 'rawblue.mean', 'rawgreen.mean', 'exred.mean', 'exblue.mean' ,
                 'exgreen.mean', 'value.mean', 'saturation.mean', 'hue.mean']
    data.loc[:, cat_cols] = get_dummies(data[cat_cols])
    ncat = learncats(data.values, classcol=-1, continuous_ids=[data.columns.get_loc(c) for c in cont_cols])
    return data.values.astype(float), ncat


def german(data):
    cat_cols = [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19, 20]
    cont_cols = [1, 4, 7, 10, 12, 15, 17]
    data.iloc[:, cat_cols] = get_dummies(data[cat_cols])
    ncat = learncats(data.values, classcol=-1, continuous_ids=cont_cols)
    return data.values.astype(float), ncat


def vowel(data):
    cat_cols = ['Speaker_Number', 'Sex', 'Class']
    data.loc[:, cat_cols] = get_dummies(data[cat_cols])
    ncat = learncats(data.values, classcol=data.shape[1]-1)
    return data.values.astype(float), ncat


def cmc(data):
    cat_cols = ['Wifes_education', 'Husbands_education', 'Wifes_religion', 'Wifes_now_working%3F',
            'Husbands_occupation', 'Standard-of-living_index', 'Media_exposure', 'Contraceptive_method_used']
    cont_cols = ['Wifes_age', 'Number_of_children_ever_born']
    data.loc[:, cat_cols] = get_dummies(data[cat_cols])
    ncat = learncats(data.values, classcol=data.shape[1]-1)
    return data.values.astype(float), ncat

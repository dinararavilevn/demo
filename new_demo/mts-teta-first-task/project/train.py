import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_absolute_error

import lightgbm as lgbm

from prep import read_data, clean_data,prep_date, add_feature, label_transformer, split_data


def train_lgbm(X_train, y_train, X_test, y_test):
    params = {'bagging_fraction': 0.9615087447098526,
              'bagging_freq': 0,
              'boosting_type': 'gbdt',
              'colsample_bytree': None,
              'feature_fraction': 0.8456892704201968,
              'lambda_l1': 0.24842029273784616,
              'lambda_l2': 0.10134215415038501,
              'learning_rate': 0.08006161308805573,
              'metric': 'mae',
              'min_child_samples': None,
              'min_child_weight': 0.007497551098352097,
              'min_data_in_leaf': 150,
              'min_sum_hessian_in_leaf': None,
              'num_leaves': 152,
              'objective': 'regression',
              'seed': 42,
              'subsample_for_bin': 3500,
              'verbose': -1,
              'n_estimators': 5000}
    train_dataset = lgbm.Dataset(X_train, y_train,
                                 categorical_feature=['building_type', 'object_type', 'region', 'year'])
    test_dataset = lgbm.Dataset(X_test, y_test,
                                categorical_feature=['building_type', 'object_type', 'region', 'year'])
    model = lgbm.train(params=params,
                       train_set=train_dataset,
                       valid_sets=[train_dataset, test_dataset],
                       num_boost_round=20000,
                       early_stopping_rounds=50,
                       verbose_eval=100)
    with open("model/lgbm.pkl", "wb") as file:
        pickle.dump(model, file)


if __name__ == "__main__":
    df = read_data("data/all_v2.csv")
    df = clean_data(df)
    df = add_feature(df)
    df = prep_date(df)
    df = label_transformer(df)
    print(df.columns)
    #train, test = split_data(df)
    #train_lgbm(train.drop("price", axis=1), train.price,
    #           test.drop("price", axis=1), test.price)

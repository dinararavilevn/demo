import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_absolute_error


def read_data(path):
    return pd.read_csv(path)


def clean_data(data):
    data = data.query('price > 100000')
    data = data.query('price < 500000000')
    data = data.query('rooms != -2')
    data.index = np.arange(data.shape[0])
    return data


def prep_date(data):
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
    data = data.sort_values(by='date')
    data['year'] = data['date'].dt.year
    data = data.drop(["date", "time"], axis=1)
    return data


def add_feature(data):
    def calc_mean_room_area():
        return (data['area'] - data['kitchen_area']) / (abs(data['rooms']))

    data['mean_room_area'] = calc_mean_room_area()
    data['percent_of_kitchen_area'] = data['kitchen_area'] / data['area']
    data['percent_of_level'] = data['level'] / data['levels']
    return data


def label_transformer(data):
    categoricals = ['building_type', 'object_type', 'region', 'year']
    for col in categoricals:
        le = LabelEncoder()
        le.fit(data[col])
        data[col] = le.transform(data[col])
        with open(f"encoder/label_{col}.pkl", "wb") as file:
            pickle.dump(le, file)
    return data


def split_data(data):
    train = data.iloc[data.index[:-100000]]
    test = data.iloc[data.index[-100000:]]
    return train, test

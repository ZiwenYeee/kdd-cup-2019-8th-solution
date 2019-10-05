import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys

def get_datetime(train, col, f = "%Y-%m-%d %H:%M:%S"):
    train_fe = pd.DataFrame([])
    s = pd.to_datetime(train[col])
    train_fe[col + '_dayofweek'] = s.dt.dayofweek
    train_fe[col + '_weekend'] = train_fe[col + '_dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    train_fe[col + '_hour'] = s.dt.hour
    return train_fe

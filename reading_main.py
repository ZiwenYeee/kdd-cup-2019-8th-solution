
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import os

def reading_main(data_path):
    train_queries_1 = pd.read_csv(data_path + 'train_queries_phase1.csv')
    train_plans_1 = pd.read_csv(data_path + 'train_plans_phase1.csv')
    train_plans_1['plans'] = train_plans_1['plans'].apply(lambda x: json.loads(x))
    train_clicks_1 = pd.read_csv(data_path + 'train_clicks_phase1.csv')

    train_queries_2 = pd.read_csv(data_path + 'train_queries_phase2.csv')
    train_plans_2 = pd.read_csv(data_path + 'train_plans_phase2.csv')
    train_plans_2['plans'] = train_plans_2['plans'].apply(lambda x: json.loads(x))
    train_clicks_2 = pd.read_csv(data_path + 'train_clicks_phase2.csv')

    profiles = pd.read_csv(data_path +'profiles.csv')

    test_queries = pd.read_csv(data_path + 'test_queries.csv')
    test_plans = pd.read_csv(data_path + 'test_plans.csv')
    test_plans['plans'] = test_plans['plans'].apply(lambda x: json.loads(x))
    return train_queries_1, train_plans_1, train_queries_2, train_plans_2, train_clicks_1,\
         train_clicks_2, profiles, test_queries, test_plans

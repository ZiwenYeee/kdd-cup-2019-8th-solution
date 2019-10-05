import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from clicks.clicks_feat import clicks_feat
from queries.queries_feat import queries_feat
from plans.plans_split_feat import plans_split_feat
from Basic_function import timer
import warnings
warnings.filterwarnings('ignore')

def feature_engineer(train, test, clicks, tr_q, ts_q, tr_p, ts_p, pro, p1 = False, p2 = True, p3 = True):
    print("feature_engineer part")
    if p1:
        with timer("clicks part:-==========="):
            train, test = clicks_feat(train, test, clicks, tr_q, ts_q, tr_p, ts_p)
    if p2:
        with timer("queries part:==========="):
            train, test = queries_feat(train, test, tr_p, tr_q, ts_p, ts_q, pro)
    if p3:
        with timer("plans part:============="):
            train, test = plans_split_feat(train, test, tr_p, ts_p, tr_q, ts_q)
    with timer("merge full dataset:"):
        train = pd.merge(train, clicks[['sid','click_mode','click_time']], on = ['sid'], how = 'left')
        train['click_mode'] = train['click_mode'].fillna(0).astype(int)
        train['click_time'].fillna(train['req_time'], inplace = True)
        train.fillna(0, inplace = True)
        test.fillna(0, inplace = True)
        features = [col for col in train.columns \
            if col not in ['sid','o', 'd',
                           'click_time','click_mode',
                           'req_time','plan_time','plans']]


    return train, test, features

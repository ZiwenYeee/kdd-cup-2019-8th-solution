import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import time
from Basic_function import timer




def voting_abnormal(tr, ts, clicks, tr_q, ts_q, tr_p, ts_p):
    tr_predict = pd.read_csv('D:\\2019KDD\\sub\\cnt_train_cv_6901_voting_23.csv')
    ts_predict = pd.read_csv('D:\\2019KDD\\sub\\cnt_test_cv_6901_voting_23.csv')
    tr[tr_predict.columns] = tr_predict
    ts[ts_predict.columns] = ts_predict
    tr = tr.loc[tr['cnt'] < 23,['sid','req_time']].reset_index(drop = True)
    ts = ts.loc[ts['cnt'] < 23,['sid','req_time']].reset_index(drop = True)
    tr_q = pd.merge(tr[['sid']], tr_q, on = ['sid'], how = 'left')
    tr_p = pd.merge(tr[['sid']], tr_p, on = ['sid'], how = 'left')
    ts_q = pd.merge(ts[['sid']], ts_q, on = ['sid'], how = 'left')
    ts_p = pd.merge(ts[['sid']], ts_p, on = ['sid'], how = 'left')
    clicks = pd.merge(tr[['sid']], clicks, on = ['sid'], how = 'left')
    return tr, ts, clicks, tr_q, ts_q, tr_p, ts_p
# main, test, train_clicks, train_queries, test_queries, train_plans, test_plans = voting_abnormal(main, test,
#                                                                                                  train_clicks, train_queries,
#                                                                                                  test_queries, train_plans, test_plans)

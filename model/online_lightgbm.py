import numpy as np
import pandas as pd
import gc
import time
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
warnings.simplefilter(action='ignore')
def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    preds = preds.reshape(12, -1).T
    preds = preds.argmax(axis = 1)
    f_score = f1_score(labels , preds,  average = 'weighted')
    return 'f1_score', f_score, True

def linear_split_v1(X, feature):
    X['req_time'] = pd.to_datetime(X['req_time'])

    X_train = X.loc[X['req_time'] < '2018-11-07', feature].reset_index(drop = True)

    X_test = X.loc[(X['req_time'] > '2018-11-07') &
                    (X['req_time'] < '2018-11-23'), feature].reset_index(drop = True)

    X_valid = X.loc[(X['req_time'] > '2018-11-23') &
                    (X['req_time'] < '2018-12-01'), feature].reset_index(drop = True)

    y_train = X.loc[X['req_time'] < '2018-11-07', 'click_mode'].reset_index(drop = True)

    y_test = X.loc[(X['req_time'] > '2018-11-07') &
                    (X['req_time'] < '2018-11-23'), 'click_mode'].reset_index(drop = True)

    y_valid = X.loc[(X['req_time'] > '2018-11-23') &
                    (X['req_time'] < '2018-12-01'), 'click_mode'].reset_index(drop = True)
    y_train = list(y_train)
    y_test = list(y_test)
    y_valid = list(y_valid)
    return X_train, X_test, X_valid, y_train, y_test, y_valid

def linear_split_v2(X, feature):
    X['req_time'] = pd.to_datetime(X['req_time'])

    X_train = X.loc[X['req_time'] < '2018-11-23', feature].reset_index(drop = True)

    X_valid = X.loc[(X['req_time'] > '2018-11-23') &
                    (X['req_time'] < '2018-12-01'), feature].reset_index(drop = True)

    y_train = X.loc[X['req_time'] < '2018-11-23', 'click_mode'].reset_index(drop = True)

    y_valid = X.loc[(X['req_time'] > '2018-11-23') &
                    (X['req_time'] < '2018-12-01'), 'click_mode'].reset_index(drop = True)
    y_train = list(y_train)
    y_valid = list(y_valid)
    return X_train, X_valid, y_train, y_valid

def online_lightgbm(x_train,x_test, feature, category_ = 'auto'):
    ntrain = x_train.shape[0]
    ntest = x_test.shape[0]
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_2 = np.zeros((ntest,))
    feature_importance_df = pd.DataFrame()
    class_num = x_train['click_mode'].nunique()
    params = {
    'task':'train',
    'boosting_type':'gbdt',
    'objective':'multiclass',
    'num_class': class_num,
    'metric': 'None',
    # 'boosting_type':'goss',
    'learning_rate': 0.1,
    'num_leaves': 200,

    'feature_fraction': 0.5,
    'subsample':0.5,

    'num_threads': -1,
    'seed': 1,
    # 'max_bin':127,
    'reg_alpha': 1,
    'reg_lambda': 1,
    }

    X_train, X_test, y_train, y_test = linear_split_v2(x_train, feature)
    dtrain = lgb.Dataset(data = X_train,
                        label = y_train,
                        free_raw_data = False, silent = True)
    dtest = lgb.Dataset(data = X_test,
                        label = y_test,
                        free_raw_data = False, silent = True)   
    
    # sample_weight = dict(x_train.shape[0]/(12*x_train.groupby(['click_mode'])['sid'].count()) )
    # dtrain.set_weight(pd.Series(y_train).map(sample_weight) )
    # dtest.set_weight(pd.Series(y_test).map(sample_weight) )
    
    clf = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round = 2000,
            valid_sets=[dtrain, dtest],
            early_stopping_rounds = 50,\
            feval = evalerror,
            verbose_eval=50,
            categorical_feature= category_
        )

    sample_weight = dict(x_train.shape[0]/(12*x_train.groupby(['click_mode'])['sid'].count()) )
    dtrain.set_weight(pd.Series(y_train).map(sample_weight) )
    dtest.set_weight(pd.Series(y_test).map(sample_weight) )
    
    clf = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round = 2000,
            valid_sets=[dtest],
            early_stopping_rounds = 15,
            init_model = clf,
            feval = evalerror,
            verbose_eval=5
        )

    
    bst_iter = clf.best_iteration
    lgb_valid = clf.predict(dtest.data)
    lgb_test = clf.predict(x_test[feature])
#     train_prediction = clf.predict(dtest.data)
#     oof_train = train_prediction
#     oof_valid = lgb_prediction
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = feature
    fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')
    fold_importance_df["fold"] = 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    print('one week for validation strategy,last week F1 : %.6f' %(f1_score(y_test,
    lgb_valid.argmax(axis = 1), average = 'weighted') ) )
    del clf, dtrain, dtest
    gc.collect()
    return lgb_valid, lgb_test, feature_importance_df

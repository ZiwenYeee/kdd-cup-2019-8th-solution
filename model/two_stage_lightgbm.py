import numpy as np
import pandas as pd
import gc
import time
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    preds = preds.reshape(12, -1).T
    preds = preds.argmax(axis = 1)
    f_score = f1_score(labels , preds,  average = 'weighted')
    return 'f1_score', f_score, True

def linear_split(X, feature):
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

def two_stage_lightgbm(x_train,x_test, feature):
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
    X_train, X_valid, y_train, y_valid = linear_split(x_train, feature)

    oof_train_2 = np.zeros((X_valid.shape[0],))
    dtrain = lgb.Dataset(data = X_train,
                        label = y_train,
                        free_raw_data = False, silent = True)
    dtest = lgb.Dataset(data = X_valid,
                        label = y_valid,
                        free_raw_data = False, silent = True)

    clf = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=2000,
            valid_sets=[dtest],
            early_stopping_rounds=50,
            feval = evalerror,
            verbose_eval=50
        )
    bst_iter = clf.best_iteration
    lgb_prediction = clf.predict(dtest.data)
    # lgb_prediction = lgb_prediction.argmax(axis = 1)
    lgb_test = clf.predict(x_test[feature])
    # lgb_test = lgb_test.argmax(axis = 1)
    oof_train_2 = lgb_prediction
    oof_test_2 = lgb_test
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = feature
    fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')
    fold_importance_df["fold"] = 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    print('F1 : %.6f' % (f1_score(dtest.label, oof_train_2.argmax(axis = 1), average = 'weighted') ) )
    del clf, dtrain, dtest
    gc.collect()
    # return oof_train_2, oof_test_2, feature_importance_df

    total = lgb.Dataset(data = x_train[feature],
                        label = x_train['click_mode'],
                        free_raw_data = False, silent = True)
    clf = lgb.train(
        params=params,
        train_set=total,
        num_boost_round = int(bst_iter),
            valid_sets=[total],
            early_stopping_rounds=200,
            feval = evalerror,
            verbose_eval=200
        )
    total_prediction = clf.predict(total.data)
    # total_prediction = total_prediction.argmax(axis = 1)
    lgb_test_total = clf.predict(x_test[feature])
    # lgb_test_total = lgb_test_total.argmax(axis = 1)
    oof_train = total_prediction
    oof_test = lgb_test_total
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = feature
    fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')
    fold_importance_df["fold"] = 2
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    print("Full F1 score %.6f" % f1_score(x_train['click_mode'], oof_train.argmax(axis = 1), average = 'weighted'))
    return oof_train, oof_test,oof_train_2, oof_test_2, feature_importance_df

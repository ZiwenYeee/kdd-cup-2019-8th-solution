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
    return X_train, X_valid, y_train, y_valid

def cv_validation_lightgbm(x_train, x_test, feature,num_folds, stratified = False):
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
    'subsample':0.8,

    'num_threads': -1,
    'seed': 1,
    # 'max_bin':127,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    }
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=777)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=777)


    X_train, X_valid, y_train, y_valid = linear_split(x_train, feature)

    ntrain, nvalid, ntest = X_train.shape[0], X_valid.shape[0], x_test.shape[0]
    oof_train = np.zeros((ntrain, class_num))
    oof_valid = np.zeros((nvalid, class_num))
    oof_test = np.zeros((ntest, class_num))
    oof_test_list = []
    oof_valid_list = []
    feature_importance_df = pd.DataFrame()
    score_list = []
    valid_list = []
    for n_fold, (train_idx, test_idx) in enumerate(folds.split(X_train[feature], y_train)):
        dtrain = lgb.Dataset(data = X_train[feature].iloc[train_idx],
                            label = y_train.loc[train_idx],
                            free_raw_data = False, silent = True)
        dtest = lgb.Dataset(data = X_train[feature].loc[test_idx],
                           label = y_train.loc[test_idx],
                           free_raw_data = False, silent = True)

        clf = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=2000,
            valid_sets=[dtest],
            early_stopping_rounds=200,
            feval = evalerror,
            verbose_eval=50
        )

        # oof_train
        lgb_prediction = clf.predict(dtest.data)
        oof_train[test_idx, :] = lgb_prediction

        # oof_test
        lgb_test = clf.predict(x_test[feature])
        oof_test += lgb_test
        oof_test_list.append(lgb_test)

        # oof_valid
        lgb_valid = clf.predict(X_valid[feature])
        oof_valid += lgb_valid
        oof_valid_list.append(lgb_valid)

        # feature importance
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feature
        fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        # score
        score = f1_score(dtest.label, oof_train[test_idx].argmax(axis = 1), average = 'weighted')
        valid_score = f1_score(y_valid, oof_valid.argmax(axis = 1), average = 'weighted')
        score_list.append(score)
        valid_list.append(valid_score)
        print('Fold %2d F1 : %.6f\n' % (n_fold + 1, score))
        print('Last week validation %2d F1 : %.6f\n' % (n_fold + 1, valid_score))
        del clf, dtrain, dtest
        gc.collect()
    print("###### Full F1 score %.6f" % f1_score(y_train,
                                                oof_train.argmax(axis = 1),
                                                average = 'weighted'))
    print("last week averaged Full F1 score %.6f"% f1_score(y_valid,
                                                            oof_valid.argmax(axis = 1),
                                                            average = 'weighted'))

    oof_test = np.argmax(oof_test, axis=1)

    # print('score_list', score_list)
    print('score_list_mean', np.mean(score_list))
    print('last week mean score:', np.mean(valid_list))
    return oof_train, oof_test, oof_valid, oof_valid_list, oof_test_list, feature_importance_df

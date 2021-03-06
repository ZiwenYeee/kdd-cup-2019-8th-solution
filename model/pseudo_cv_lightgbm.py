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

def pseudo_cv_lightgbm(x_train, y_train, x_test, fake, feature,num_folds, stratified = False):
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

    ntrain, ntest = x_train.shape[0], x_test.shape[0]
    oof_train = np.zeros((ntrain, class_num))
    oof_test = np.zeros((ntest, class_num))
    oof_test_list = []

    feature_importance_df = pd.DataFrame()
    score_list = []
    for n_fold, (train_idx, test_idx) in enumerate(folds.split(x_train[feature], y_train)):
        _x_train, _y_train = x_train[feature].iloc[train_idx], y_train.loc[train_idx]
        _x_valid, _y_valid = x_train[feature].iloc[test_idx], y_train.loc[test_idx]
        _x_train = pd.concat([_x_train, fake[feature]]).reset_index(drop = True)
        _y_train = pd.concat([_y_train, fake['click_mode']]).reset_index(drop = True)
        dtrain = lgb.Dataset(data = _x_train,
                            label = _y_train,
                            free_raw_data = False, silent = True)
        dtest = lgb.Dataset(data = _x_valid,
                           label = _y_valid,
                           free_raw_data = False, silent = True)

        clf = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=10000,
            valid_sets=[dtest],
            early_stopping_rounds=200,
            feval = evalerror,
            verbose_eval=50
        )

        # oof_train
        lgb_prediction = clf.predict(dtest.data)
        oof_train[test_idx,:] = lgb_prediction

        # oof_test
        lgb_test = clf.predict(x_test[feature])
        oof_test += lgb_test
        oof_test_list.append(lgb_test)

        # feature importance
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feature
        fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        # score
        score = f1_score(dtest.label, oof_train[test_idx].argmax(axis = 1), average = 'weighted')
        score_list.append(score)
        print('Fold %2d F1 : %.6f\n' % (n_fold + 1, score))
        del clf, dtrain, dtest
        gc.collect()
    print("###### Full F1 score %.6f" % f1_score(y_train, oof_train.argmax(axis = 1), average = 'weighted'))

    oof_test = np.argmax(oof_test, axis=1)

    print('score_list', score_list)
    print('score_list_mean', np.mean(score_list))
    return oof_train, oof_test, oof_test_list, feature_importance_df

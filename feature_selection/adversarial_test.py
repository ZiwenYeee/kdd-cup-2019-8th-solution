import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
def adversarial_test(train, test, feature):
    y1 = np.array([0]*train.shape[0])
    y2 = np.array([1]*test.shape[0])
    y = np.concatenate((y1, y2))

    X_data = pd.concat([train[feature], test[feature]])
    X_data.reset_index(drop=True, inplace=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)

    lgb_model = lgb.LGBMClassifier(max_depth=-1,
                                   n_estimators=100,
                                   learning_rate=0.1,
                                   objective='binary',
                                   n_jobs=-1)

    counter = 1
    score_list = []
    lgb_imp = pd.DataFrame([])
    for train_index, test_index in skf.split(X_data, y):
        print('\nFold {}'.format(counter))
        X_fit, X_val = X_data.loc[train_index], X_data.loc[test_index]
        y_fit, y_val = y[train_index], y[test_index]

        lgb_model.fit(X_fit, y_fit, eval_metric='auc',
              eval_set=[(X_val, y_val)],
              verbose=100, early_stopping_rounds=10)
        predict = lgb_model.predict_proba(X_val)[:,1]
        lgb_imp = pd.DataFrame(lgb_model.feature_importances_, feature, ['values']).sort_values(['values'],ascending = False)
        score = roc_auc_score(y_val, predict)
        score_list.append(score)
        counter+=1
    return lgb_imp

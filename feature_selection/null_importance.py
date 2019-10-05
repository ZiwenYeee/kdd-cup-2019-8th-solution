import numpy as np
import pandas as pd
import gc
import time
import lightgbm as lgb
from sklearn.metrics import f1_score

def null_importance(train_df, train_features, nb_runs = 20):
    def get_feature_importances(data, train_features, shuffle = False):
        y = data['click_mode'].copy()
        if shuffle:
        # Here you could as well use a binomial distribution
            y = data['click_mode'].copy().sample(frac=1.0)
        dtrain = lgb.Dataset(data[train_features], y, free_raw_data=False, silent=False)
        params = {
        'task':'train',
        'boosting_type':'gbdt',
        'objective':'multiclass',
        'num_class': 12,
        'metric': 'None',
        'learning_rate': 0.1,
        'num_leaves': 61,
        'feature_fraction': 0.8,
        'subsample': 1,
        'num_threads': -1,
        'seed': 2019,
        'reg_alpha': 0,
        'reg_lambda': 0.1,
        'verbosity': -1
        }
        clf = lgb.train(params=params,train_set=dtrain,num_boost_round=30)
        preds = clf.predict(data[train_features])
        preds = preds.argmax(axis = 1)
        # Get feature importances
        imp_df = pd.DataFrame()
        imp_df["feature"] = list(train_features)
        imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
        imp_df["importance_split"] = clf.feature_importance(importance_type='split')
        imp_df['trn_score'] = f1_score(y, preds,average='weighted')

        return imp_df
    np.random.seed(817)
    # Get the actual importance, i.e. without shuffling
    actual_imp_df = get_feature_importances(data = train_df, train_features = train_features, shuffle=False)
    null_imp_df = pd.DataFrame()
    start = time.time()
    dsp = ''
    for i in range(nb_runs):
    # Get current run importances
        imp_df = get_feature_importances(data=train_df, train_features = train_features, shuffle=True)
        imp_df['run'] = i + 1
    # Concat the latest importances with the old ones
        null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
    # Erase previous message
        for l in range(len(dsp)):
            print('\b', end='', flush=True)
    # Display current run and time used
    spent = (time.time() - start) / 60
    dsp = 'Done with %4d of %4d (Spent %5.1f min)' % (i + 1, nb_runs, spent)
    print(dsp, end='', flush=True)
    feature_scores = []
    for _f in actual_imp_df['feature'].unique():
        f_null_imps_gain = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
        f_act_imps_gain = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].mean()

        gain_score = np.log(1e-10 + f_act_imps_gain / (1 + np.percentile(f_null_imps_gain, 75)))  # Avoid didvide by zero
        f_null_imps_split = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
        f_act_imps_split = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].mean()
        split_score = np.log(1e-10 + f_act_imps_split / (1 + np.percentile(f_null_imps_split, 75)))  # Avoid didvide by zero
        feature_scores.append((_f, split_score, gain_score))

    scores_df = pd.DataFrame(feature_scores, columns=['feature', 'split_score', 'gain_score'])
    new_list = scores_df.sort_values(by=['gain_score'],ascending=False).reset_index(drop=True)
    return new_list

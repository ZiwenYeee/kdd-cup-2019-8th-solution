
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import time
from Basic_function import timer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def get_fe_plans_svd(train, test, plans, test_plans, prin = 10):
    p1 = plans[['sid', 'plans']]
    p2 = test_plans[['sid', 'plans']]
    p = pd.concat([p1, p2]).reset_index(drop = True)

    def plans_count(x):
        mode_list = []
        mode_texts = []
        try:
            for i in x:
                mode_list.append(i['transport_mode'])
            mode_texts.append(
            ' '.join(['word_{}'.format(mode) for mode in mode_list]))
            return mode_texts
        except:
            mode_texts.append('word_null')
            return mode_texts

    def svd_generation(plans):
        data = plans['plans'].apply(lambda x: plans_count(x))
        q = []
        for i in range(len(data)):
            q.append(data[i][0])
        return q

    d = svd_generation(p)
    tfidf_enc = TfidfVectorizer(ngram_range=(1, 2))
    tfidf_vec = tfidf_enc.fit_transform(d)
    svd_enc = TruncatedSVD(n_components=prin, n_iter=20, random_state=2019)
    mode_svd = svd_enc.fit_transform(tfidf_vec)
    mode_svd = pd.DataFrame(mode_svd)
    mode_svd.columns = ['svd_mode_{}'.format(i) for i in range(prin)]
    p[mode_svd.columns] = mode_svd
    p.drop(['plans'], axis = 1, inplace = True)
    train = pd.merge(train, p, on = ['sid'], how = 'left')
    test  = pd.merge(test,  p, on = ['sid'], how = 'left')
    return train, test

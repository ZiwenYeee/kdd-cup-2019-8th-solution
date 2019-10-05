import geohash2 as geohash
import sys
import pandas as pd
import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

def get_fe_encode_geo_time(query):
      
    o_geohash = np.reshape(query['o_geohash'].values,(-1,1))
    d_geohash = np.reshape(query['d_geohash'].values,(-1,1))
    # pid = np.reshape(query['pid'].values, (-1,1))
    # features = np.hstack((o_geohash, d_geohash, pid)).tolist()
    # tfidf_enc = TfidfVectorizer(ngram_range=(1, 3))
    # features = [' '.join([str(mode) for mode in mode_list]) for mode_list in features]
    dayofweek = np.reshape(query['req_time_dayofweek'].values,(-1,1))
    hour = np.reshape(query['req_time_hour'].values,(-1,1))
    features = np.hstack((o_geohash, d_geohash, dayofweek, hour)).tolist()
    features = [' '.join([str(mode) for mode in mode_list]) for mode_list in features]
    tfidf_enc = TfidfVectorizer(ngram_range=(1, 4))
    features_vec = tfidf_enc.fit_transform(features)
    svd = TruncatedSVD(n_components= 10, n_iter = 5, random_state=2019)
    new_features = svd.fit_transform(features_vec)
    col = np.shape(new_features)[1]
    features_name = ['embed_queries_feature_' + str(i+1) for i in range(col)]
    new_features = pd.DataFrame(new_features,columns = features_name)
    return new_features
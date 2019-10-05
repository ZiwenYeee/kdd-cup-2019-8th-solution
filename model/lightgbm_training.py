import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore')
from Basic_function import timer

from model.last_week_lightgbm import last_week_lightgbm
from model.kfold_lightgbm import kfold_lightgbm
from model.cv_validation_lightgbm import cv_validation_lightgbm
from model.pseudo_cv_lightgbm import pseudo_cv_lightgbm
from model.pseudo_last_week_lightgbm import pseudo_last_week_lightgbm

def lightgbm_training(train, test, feature, strategy = "cv"):
    if strategy == 'cv':
        with timer("cross validation strategy:"):
            train_label = train['click_mode'].astype(int)
            oof_train, oof_test, oof_test_list, df_imp = kfold_lightgbm(train, train_label,
                                                    test, feature, 5, stratified = True)
            return oof_train, oof_test_list, df_imp
    if strategy == 'last_week':
        with timer("last week strategy:"):
            oof_valid, oof_test, feature_importance_df = last_week_lightgbm(train, test, feature)
            return oof_valid, oof_test, feature_importance_df
    if strategy == 'both':
        with timer("corss training - last week strategy:"):
            oof_train, oof_test, oof_valid, oof_valid_list, oof_test_list, feature_importance_df = \
            cv_validation_lightgbm(train, test, feature, 5)
            return oof_train, oof_valid_list, feature_importance_df

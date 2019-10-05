from sklearn.metrics import f1_score
import gc, os, sys, datetime, time
from sklearn.model_selection import StratifiedKFold
import gc
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')

class G_model:
    def __init__(self):
        self.weight_list = []
        self.score_train_list = []
        self.score_valid_list = []
        self.best_iteration = 0

    def predict(self, oof_test):
        return oof_test * self.weight_list[self.best_iteration]
    
    def fit(self, dtrain, dvalid, dtest = None,
            n_search = 1000, early_stopping_rounds = 10, verbose = 1, seed = 233):
        
        oof_train, train_label = dtrain[0], dtrain[1]
        oof_valid, valid_label = dvalid[0], dvalid[1]
        if not dtest is None:
            oof_test, test_label = dtest[0], dtest[1]

        np.random.seed(seed)
        weight_num = 12
        weight = np.array([1.0] * weight_num)

        weight_list = []
        score_train_list, score_valid_list, score_test_list = [], [], []

        pred_old = oof_train
        counter = 0
        for i in range(n_search):
            score_old = f1_score(train_label, pred_old.argmax(axis = 1), average = 'weighted')
            score_new_list = []
            del_weight_list = []
            for j in range(12):
                del_weight = np.zeros(12)
                del_weight[j] = 1 * np.random.normal(loc = 0.0, scale = 1.0)
                del_weight_list.append(del_weight)
                new_weights = weight + del_weight

                # score at train
                pred_new = oof_train * new_weights
                score = f1_score(train_label, pred_new.argmax(axis = 1), average = 'weighted')
                score_new_list.append(score)

            idx_max = np.argmax(score_new_list)
            new_weights = weight + del_weight_list[idx_max]
            pred_new = oof_train * new_weights   
            score_valid = f1_score(valid_label, (oof_valid * new_weights).argmax(axis = 1), average = 'weighted')
            if not dtest is None:
                score_test = f1_score(test_label, (oof_test * new_weights).argmax(axis = 1), average = 'weighted')

            if score_new_list[idx_max] - score_old > -0.00001 * np.random.random():
                weight = new_weights
                pred_old = pred_new

                weight_list.append(weight)
                score_train_list.append(score_new_list[idx_max])
                score_valid_list.append(score_valid)
                if not dtest is None:
                    score_test_list.append(score_test)
                
                idx_cur_max = np.argmax(score_valid_list)
                
                if counter % verbose == 0:
                    if not dtest is None:
                        print('[***] i={:03d}, cnt={:03d}, best={:03d}, train={:.6f}, valid={:.6f}, test={:.6f}'.\
                              format(i, counter, idx_cur_max, score_train_list[-1], score_valid, score_test))
                    else:
                        print('[***] i={:03d}, cnt={:03d}, best={:03d}, train={:.6f}, valid={:.6f}'.\
                              format(i, counter, idx_cur_max, score_train_list[-1], score_valid))

                # for early_stopping
                if counter - idx_cur_max >= early_stopping_rounds:
                    break
                counter += 1
            
        print('[+++] best')
        if not dtest is None:
            print('[+++] i={:03d}, cnt={:03d}, best={:03d}, train={:.6f}, valid={:.6f}, test={:.6f}'.\
                  format(0, 0, idx_cur_max,
                         score_train_list[idx_cur_max],
                         score_valid_list[idx_cur_max],
                         score_test_list[idx_cur_max],))
        else:
            print('[+++] i={:03d}, cnt={:03d}, best={:03d}, train={:.6f}, valid={:.6f}'.\
                  format(0, 0, idx_cur_max,
                         score_train_list[idx_cur_max],
                         score_valid_list[idx_cur_max]))
            
        self.weight_list = weight_list
        self.score_train_list = score_train_list
        self.score_valid_list = score_valid_list
        self.best_iteration = idx_cur_max
        
        
def greedy_oof(train, train_label, test, test_label = None, n_kfolds = 5):
    n_train, n_test = train.shape[0], test.shape[0]
    class_num = 12
    oof_train = np.zeros((n_train, class_num))
    oof_test = np.zeros((n_test, class_num))
    score_list = []
    
    model_list = []
    skf = StratifiedKFold(n_splits = n_kfolds, shuffle = True, random_state = 777).split(train, train_label) 
    for i, (train_idx, valid_idx) in enumerate(skf):
        print('############################################################ fold = {}'.format(i))
        print('####### cur time = ' + str(datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")))
        
        X_train, y_train = train[train_idx], train_label[train_idx]
        X_valid, y_valid = train[valid_idx], train_label[valid_idx]
        dtrain = [X_train, y_train]
        dvalid = [X_valid, y_valid]
        dtest = None
        if not test_label is None:
            dtest = [test, test_label]
        
        model = G_model()
        model.fit(dtrain,
                  dvalid,
                  dtest,
                  n_search = 1000,
                  early_stopping_rounds = 5,
                  verbose = 1)

        oof_train[valid_idx,:] = model.predict(X_valid)
        oof_test += model.predict(test) / n_kfolds
        score_list.append(f1_score(y_valid, oof_train[valid_idx].argmax(axis = 1), average = 'weighted'))
        model_list.append(model)
        print('')
        
    print('score_list', score_list)
    print('score_list_mean = {:.6f}'.format(np.mean(score_list)))
    print("score full train = {:.6f}".format(f1_score(train_label, oof_train.argmax(axis = 1), average = 'weighted')))
    gc.collect()
    return oof_train, oof_test, model_list
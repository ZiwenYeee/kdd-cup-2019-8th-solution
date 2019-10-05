import multiprocessing as mul
import os
import gc
import sys
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score

def selected_voting(oof_train, train_q, iteration, num, k):
    w_list, oof_train_pred_list, score_list = parallel_voting(oof_train, train_q, iteration, num)
    idx_sel = list(range(k))
    idx_sel_score = [0] * k
    y_train = train_q['lable'].astype(int)

    def voting_2(y_list):
        y_all = np.vstack([y_list]).T
        from scipy import stats
        y = stats.mode(y_all, axis = 1)[0].flatten()
        y_cnt = stats.mode(y_all, axis = 1)[1].flatten()
        return y, y_cnt
    while True:
        score_try = []
        for i in range(len(score_list)):
            if i in set(idx_sel):
                continue
            idx_try = idx_sel + [i]
            voting_y, voting_y_cnt = voting_2([oof_train_pred_list[idx] for idx in idx_try])
            voting_score = f1_score(y_train, voting_y, average = 'weighted')
            score_try.append([i, voting_score])
        score_try = pd.DataFrame(score_try)
        score_try.columns = ['idx', 'score']

        score_try = score_try.sort_values(by = 'score', ascending = False).reset_index(drop = True)
        print('score_try top 1')
        print(score_try.head(1))

    # add the best new one
        if len(idx_sel_score) == 0 or score_try['score'][0] > idx_sel_score[-1]:
            idx_sel += [score_try['idx'][0]]
            idx_sel_score += [score_try['score'][0]]
        else:
            print('without improvement')
            break
        print('')
    return w_list, idx_sel


def parallel_voting(oof_train, train_q, iteration, num):
    oof_train_list = [oof_train]
    oof_train_pred_list = []
    score_list = []
    voting_score_list = []
    w_list = []
    pool = mul.Pool(mul.cpu_count() - 1)
    results_list = []
    results = []
    for n in range(num):
        results_list.append(
                pool.apply_async(monte_search, args=(oof_train, train_q, iteration, n)))
    pool.close()
    pool.join()
    y_train = train_q['lable'].astype(int)
    w_list = [result.get() for result in results_list]
    oof_train_pred_list = [(oof_train * result.get()).argmax(axis = 1) for result in results_list]
    score_list = [f1_score(y_train, oof_train_pred, average = 'weighted') for oof_train_pred in oof_train_pred_list]
    return w_list, oof_train_pred_list, score_list

def monte_search(oof_train, train_q, n_search = 100, seed = 123):
    np.random.seed(seed)
    weight_num = 12
    weight = np.array([1.0,]* weight_num)

    ws=[]
    rr=[]
    result=[]
    pred_old = oof_train #bao_cat_9226583215911033_oof['pred_bao_cat_9226']
    counter=0

    for i in range(0,n_search):
        # learning rate
        new_weights = weight + np.array([0.05,]* weight_num)*np.random.normal(loc=0.0, scale=1.0, size=weight_num)
        new_weights[new_weights < 0.001]=0.001
        pred_new = oof_train * new_weights
        f1_new = f1_score(train_q['lable'].astype(int), pred_new.argmax(axis = 1), average = 'weighted')
        f1_old =  f1_score(train_q['lable'].astype(int), pred_old.argmax(axis = 1), average = 'weighted')
        #print(f1_new)
        diff = -f1_new + f1_old
        #diff = - roc_auc_score(train_y ,pred_new) +  f1_score(train_q['lable'].astype(int), oof_train.argmax(axis = 1), average = 'weighted')

        prob = min(1,np.exp(-diff*200000))
        random_prob = np.random.rand()
        if random_prob < prob:
            weight= new_weights
            pred_old=pred_new
            result.append(f1_new)
            ws.append(weight)
            rr.append(pred_new)
            counter +=1
        # if i % 100 == 0 and len(result) > 1:
        #     print(counter,result[-1],i)


    bestSC = np.max(result)
    bestWght = ws[np.argmax(result)]
    best_pred=rr[np.argmax(result)]
    bestWght = bestWght / np.sum(bestWght)
    coefs = bestWght
    gc.collect()
#     print(coefs)
    return coefs

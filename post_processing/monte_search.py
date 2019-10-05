import gc
import sys
import pandas as pd
import numpy as np

from sklearn.metrics import f1_score

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
        diff = -f1_new + f1_old

        prob = min(1,np.exp(-diff*200000))
        random_prob = np.random.rand()
        if random_prob < prob:
            weight= new_weights
            pred_old=pred_new
            result.append(f1_new)
            ws.append(weight)
            rr.append(pred_new)
            counter +=1
        if i % 100 == 0 and len(result) > 1:
            print(counter,result[-1],i)


    bestSC = np.max(result)
    bestWght = ws[np.argmax(result)]
    best_pred=rr[np.argmax(result)]
    bestWght = bestWght / np.sum(bestWght)
    coefs = bestWght
    gc.collect()
#     print(coefs)
    return coefs

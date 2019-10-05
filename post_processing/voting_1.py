import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.stats as stats

def voting_1(y_list):
    y_all = np.vstack([y_list]).T
    print(y_all.shape)

    from scipy import stats
    y = stats.mode(y_all, axis = 1)[0].flatten()
    for i in range(len(y_list)):
        print('y == y{} = {}'.format(i + 1, (y == y_list[i]).sum()))

    y_cnt = stats.mode(y_all, axis = 1)[1].flatten()
    for i in range(len(y_list)):
        print('y_cnt == {} = {}'.format(i + 1, (y_cnt == i + 1).sum()))
    return y, y_cnt

from sklearn.decomposition import TruncatedSVD
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

def post_analysis(queries, clicks, plans, result):
    def plan_diff(x,l, r_list = False, price = False, eta = False, distance = False):
        a = []
        p = []
        e = []
        d = []
        c = 0
        try:
            for j in x:
                a.append(j['transport_mode'])
                p.append(j['price'])
                e.append(j['eta'])
                d.append(j['distance'])
        except:
            a = p = e = d = []
        if r_list:
            if price:
                return p
            else:
                if eta:
                    return e
                else:
                    if distance:
                        return d
                    else:
                        return a
        else:
            try:
                a.index(l)
            except:
                return -1
            else:
                return  a.index(l)
    q = queries[['sid']]
    p = plans[['sid', 'plans']]
    c = clicks[['sid', 'click_mode']]
    main = pd.merge(q, p, on = ['sid'], how = 'left')
    main = pd.merge(main, c, on = ['sid'], how = 'left')
    main.click_mode.fillna(0, inplace = True)
    main = pd.merge(main, result, on = ['sid'], how = 'left')
    main['click_rank'] = list(map(lambda x, y: plan_diff(x,y), main['plans'], main['click_mode'] ) )
    main['result_rank'] = list(map(lambda x, y: plan_diff(x,y), main['plans'], main['recommend_mode'] ) )
    main['mode_list'] = list(map(lambda x, y: plan_diff(x,y, True), main['plans'], main['recommend_mode'] ) )
    main['price_list'] = list(map(lambda x, y: plan_diff(x,y, True, True), main['plans'], main['recommend_mode'] ) )
    main['eta_list'] = list(map(lambda x, y: plan_diff(x,y, True, False, True), main['plans'], main['recommend_mode'] ) )
    main['distance_list'] = list(map(lambda x, y: plan_diff(x,y, True, False, False, True), main['plans'], main['recommend_mode'] ) )

    main['predict_flag'] = np.where(main['click_mode'] == main['recommend_mode'], 1, 0)
    return main

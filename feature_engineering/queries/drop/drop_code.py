# from plans.get_fe_count_plan import get_fe_count_plan
# from plans.get_fe_plans_counts import get_fe_plans_counts
# from plans.get_fe_cnt_rank_ratio import get_fe_cnt_rank_ratio

# from queries.get_fe_od_count import get_fe_od_count
# from queries.get_fe_o_d import get_fe_o_d
# from queries.get_fe_pid_count import get_fe_pid_count
# from queries.get_fe_time_heat_count import get_fe_time_heat_count

    # with timer("hour wday recommend count:"):
    #     train = get_fe_count_plan(train, plans, test_plans)
    #     test = get_fe_count_plan(test, plans, test_plans)
    # with timer("get plans count feature:"):
    #     counts_feat = get_fe_plans_counts(split_main, query_main)
    #     split_feat = split_feat.merge(counts_feat, on = ['sid'], how = 'left')
    #     split_feat_test = split_feat_test.merge(counts_feat, on = ['sid'], how = 'left')

# def get_fe_plans_num_counts(queries, plan):
#     plans = plan.copy()
#     plans['recom_num'] = plans['plans'].apply(lambda x: len(x))
#     plans['recom_mode'] = plans['plans'].apply(lambda x: x[0]['transport_mode'])
#     data = pd.merge(queries, plans, on = ['sid'], how = 'left')
#     data['values'] = 1.0
#
#     data1 = queries[['sid']]
#     df = get_fe_od_plans_num(data)
#     data1 = pd.merge(data1, df, on = ['sid'], how = 'left')
#     df = get_fe_o_plans_num(data)
#     data1 = pd.merge(data1, df, on = ['sid'], how = 'left')
#     df = get_fe_d_plans_num(data)
#     data1 = pd.merge(data1, df, on = ['sid'], how = 'left')
#     df = get_fe_o_plans_num_count(data)
#     data1 = pd.merge(data1, df, on = ['sid'], how = 'left')
#     df = get_fe_d_plans_num_count(data)
#     data1 = pd.merge(data1, df, on = ['sid'], how = 'left')
#     df = get_fe_od_plans_num_count(data)
#     data1 = pd.merge(data1, df, on = ['sid'], how = 'left')
#
#     # df = get_fe_pid_plans_num(queries, plans)
#     # data = pd.merge(data, df, on = ['sid'], how = 'left')
#     return data1

# queries['o_geohash'] = queries.apply(lambda x:ph.encode(x['o_x'], x['o_y']), axis = 1)
# queries['d_geohash'] = queries.apply(lambda x:ph.encode(x['d_x'], x['d_y']), axis = 1)
# print("geohash ready")


def get_fe_od_plans_num_count(data):
    od_mode_feat = pd.pivot_table(data,index=['o', 'd'],values=['recom_num'],columns=['values'],aggfunc='sum')
    od_mode_feat.columns = ['od_plans_num_count_' + str(x[1]) for x in od_mode_feat.columns]
    od_mode_feat = od_mode_feat.fillna(0)
    origin_col = od_mode_feat
    df = pd.merge(data[['sid', 'o', 'd']], od_mode_feat, on = ['o', 'd'], how = 'left')
    df.drop(['o', 'd'], axis = 1, inplace = True)
    df.fillna(0, inplace = True)
    return df

def get_fe_d_plans_num_count(data):
    od_mode_feat = pd.pivot_table(data,index=['d'],values=['recom_num'],columns=['values'],aggfunc='sum')
    od_mode_feat.columns = ['d_plans_num_count_' + str(x[1]) for x in od_mode_feat.columns]
    od_mode_feat = od_mode_feat.fillna(0)
    origin_col = od_mode_feat
    df = pd.merge(data[['sid','d']], od_mode_feat, on = ['d'], how = 'left')
    df.drop(['d'], axis = 1, inplace = True)
    df.fillna(0, inplace = True)
    return df


def get_fe_o_plans_num_count(data):
    df = data[['sid']]
    od_mode_feat = pd.pivot_table(data,index=['o'],values=['recom_num'],columns=['values'],aggfunc='sum')
    od_mode_feat.columns = ['o_plans_num_count_' + str(x[1]) for x in od_mode_feat.columns]
    od_mode_feat = od_mode_feat.fillna(0)
    origin_col = od_mode_feat
    df = pd.merge(data[['sid','o']], od_mode_feat, on = ['o'], how = 'left')
    df.drop(['o'], axis = 1, inplace = True)
    df.fillna(0, inplace = True)
    return df


def get_fe_pid_plans_num(data):
    od_mode_feat = pd.pivot_table(data,index=['pid'],values=['values'],columns=['recom_num'],aggfunc='sum')
    od_mode_feat.columns = ['pid_plans_num_' + str(x[1]) for x in od_mode_feat.columns]
    od_mode_feat = od_mode_feat.fillna(0)
    origin_col = od_mode_feat
    df = pd.merge(data[['sid', 'pid']], od_mode_feat, on = ['pid'], how = 'left')
    df.drop(['pid'], axis = 1, inplace = True)
    df.fillna(0, inplace = True)
    return df


def get_fe_od_plans_num(data):
    od_mode_feat = pd.pivot_table(data,index=['o','d'],values=['values'],columns=['recom_num'],aggfunc='sum')
    od_mode_feat.columns = ['od_plans_num_' + str(x[1]) for x in od_mode_feat.columns]
    od_mode_feat = od_mode_feat.fillna(0)
    origin_col = od_mode_feat
    # od_sum = od_mode_feat.sum(axis = 1)
    # od_mode_feat['od_transport_mode_sum'] = od_sum
    # for col in origin_col:
    #     od_mode_feat[col + "_ratio"] = od_mode_feat[col]/od_sum
    df = pd.merge(data[['sid', 'o','d']], od_mode_feat, on = ['o','d'], how = 'left')
    df.drop(['o','d'], axis = 1, inplace = True)
    df.fillna(0, inplace = True)
    return df

def get_fe_d_plans_num(data):
    od_mode_feat = pd.pivot_table(data,index=['d'],values=['values'],columns=['recom_num'],aggfunc='sum')
    od_mode_feat.columns = ['d_plans_num_' + str(x[1]) for x in od_mode_feat.columns]
    od_mode_feat = od_mode_feat.fillna(0)
    origin_col = od_mode_feat
    # od_sum = od_mode_feat.sum(axis = 1)
    # od_mode_feat['od_transport_mode_sum'] = od_sum
    # for col in origin_col:
    #     od_mode_feat[col + "_ratio"] = od_mode_feat[col]/od_sum
    df = pd.merge(data[['sid', 'd']], od_mode_feat, on = ['d'], how = 'left')
    df.drop(['d'], axis = 1, inplace = True)
    df.fillna(0, inplace = True)
    return df

def get_fe_o_plans_num(data):
    od_mode_feat = pd.pivot_table(data,index=['o'],values=['values'],columns=['recom_num'],aggfunc='sum')
    od_mode_feat.columns = ['o_plans_num_' + str(x[1]) for x in od_mode_feat.columns]
    od_mode_feat = od_mode_feat.fillna(0)
    origin_col = od_mode_feat
    # od_sum = od_mode_feat.sum(axis = 1)
    # od_mode_feat['od_transport_mode_sum'] = od_sum
    # for col in origin_col:
    #     od_mode_feat[col + "_ratio"] = od_mode_feat[col]/od_sum
    df = pd.merge(data[['sid', 'o']], od_mode_feat, on = ['o'], how = 'left')
    df.drop(['o'], axis = 1, inplace = True)
    df.fillna(0, inplace = True)
    return df


    with timer("pid count feature:"):
        pid_count = get_fe_pid_count(queries, plans)
        df = pd.merge(df, pid_count, on = ['sid'], how = 'left')
    with timer("od_count feature:"):
        od_count = get_fe_od_count(queries, plans)
        df = pd.merge(df, od_count, on = ['sid'], how = 'left')
    with timer("o and d feature:"):
        o_d_count = get_fe_o_d(queries)
        df = pd.merge(df, o_d_count, on = ['sid'], how = 'left')
    with timer("queries time hour heat:"):
        heat = get_fe_time_heat_count(queries)
        df = pd.merge(df, heat, on = ['sid'], how = 'left')


def get_fe_od_each_mode_count(data):
    g = data.groupby(['o', 'd']).agg({k:['mean', 'sum'] for k in ["mode_flag_" + str(i) for i in range(1,12)]})
    g.columns = ["count_od_" + "_".join(col) for col in g.columns.ravel()]
    # ratio_col = [col for col in g.columns if "_mean" in col]
    # g_sum = g[ratio_col].sum(axis = 1)
    # for i in ratio_col:
    #     g[i + "_ratio"] = g[i]/g_sum
    g.reset_index(inplace = True)
    g.fillna(0, inplace = True)
    df = pd.merge(data[['sid', 'o', 'd']], g, on = ['o', 'd'], how = 'left')
    df.drop(['o', 'd'], axis = 1, inplace = True)
    return df

def get_fe_o_each_mode_count(data):
    g = data.groupby(['o']).agg({k:['mean', 'sum'] for k in ["mode_flag_" + str(i) for i in range(1,12)]})
    g.columns = ["count_o_" + "_".join(col) for col in g.columns.ravel()]
    ratio_col = [col for col in g.columns if "_mean" in col]
    # g_sum = g[ratio_col].sum(axis = 1)
    # for i in ratio_col:
    #     g[i + "_ratio"] = g[i]/g_sum
    g.reset_index(inplace = True)
    g.fillna(0, inplace = True)
    df = pd.merge(data[['sid', 'o']], g, on = ['o'], how = 'left')
    df.drop(['o'], axis = 1, inplace = True)
    return df

def get_fe_d_each_mode_count(data):
    g = data.groupby(['d']).agg({k:['mean', 'sum'] for k in ["mode_flag_" + str(i) for i in range(1,12)]})
    g.columns = ["count_d_" + "_".join(col) for col in g.columns.ravel()]
    ratio_col = [col for col in g.columns if "_mean" in col]
    # g_sum = g[ratio_col].sum(axis = 1)
    # for i in ratio_col:
    #     g[i + "_ratio"] = g[i]/g_sum
    g.reset_index(inplace = True)
    g.fillna(0, inplace = True)
    df = pd.merge(data[['sid', 'd']], g, on = ['d'], how = 'left')
    df.drop(['d'], axis = 1, inplace = True)
    return df


def loglikelihood(preds, train_data):
    labels = train_data.get_label()
    preds = 1. / (1. + np.exp(-preds))
    w = [0.06467409, 0.05022351, 0.03314106, 0.10798409, 0.16633348,
       0.04189973, 0.12999524, 0.04578104, 0.08988228, 0.0456719,
       0.10118039, 0.12323318]
    preds = w * preds
    grad = preds - labels
    hess = preds * (1. - preds)
    return grad, hess

# fobj=loglikelihood,

import pyswarms
import pandas as pd
import numpy as np
import scipy
from scipy import stats
from haversine import haversine, Unit
from pyswarms.single.global_best import GlobalBestPSO
import math
import warnings
warnings.filterwarnings("ignore")
import ast
import csv
import time
import json
from mpi4py import MPI

year = 2019

def convert(df, colname):
    list_here = df[colname].tolist()
    max_num = max(list_here)
    min_num = min(list_here)
    deno = max_num - min_num
    ans = []
    for i in list_here:
        this = ((i-min_num)/deno)*9 + 1
        ans.append(this)
    return ans

def get_table(y):
    t_temp = pd.read_csv('/home/yilunx/Desktop/new_PSO_20210616/table_' + str(y) + '.csv').drop(columns = 'Unnamed: 0')
    t_final = t_temp[['A_cbg', 'B_store', 'I_Includes_parking_lot', 
                      'C_Percentage_of_Visits_'+str(y), 'D_ Number_of_Visits_'+str(y)]]
    change = ['H_Area_of_store', 'R_Percentage_of_Visits_by_brand_'+str(y), 'G_ Distance_between_cbg_and_store', 
    'J_POI_count_where_store_is', 'K_POI_diversity_where_store_is',
             'L_Demographic_similarity']
    for i in change:
        t_final[i] = convert(t_temp, i)
        if str(y) in i:
            t_final = t_final.rename(columns = {i:i[:-5]})
    t_final['I_Includes_parking_lot'] = t_final['I_Includes_parking_lot'].replace({0.0001: 0})
    return t_final.rename(columns = {'C_Percentage_of_Visits_'+str(y): 'C_Percentage_of_Visits','D_ Number_of_Visits_'+str(y): 'D_Number_of_Visits', 'G_ Distance_between_cbg_and_store': 'G_Distance_between_cbg_and_store'})

table = get_table(year)
params = pd.read_csv('/home/yilunx/Desktop/new_PSO_20210616/PSO_7params_NYC_20210616_2018.csv')
cbgs = list(table.A_cbg.unique())
cbgs_complete = cbgs.copy()

print('all cbgs: ', len(set(cbgs)))

all_ids = table.B_store.unique().tolist()

print('Number of All stores: ', len(all_ids))

def get_rate(df, cbg):
    df_temp = df[['A_cbg', 'B_store', 'C_Percentage_of_Visits']]
    rate = df_temp[df_temp.A_cbg == cbg].drop(columns = ['A_cbg'])
    rate = rate.reset_index().rename(columns = {'index':'safegraph_place_id',rate.columns[-1]:'actual_rate'})
    return rate

def get_params(p_df, cbg):
    df_here = p_df[p_df['cbg'] == cbg]
    variables = ['H_Area_of_store', 'I_Includes_parking_lot', 'R_Percentage_of_Visits_by_brand', 'J_POI_count_where_store_is', 'K_POI_diversity_where_store_is', 'L_Demographic_similarity', 'G_Distance_between_cbg_and_store']
    variables_dict = {}
    for i in variables:
        value_here = df_here[i].tolist()[0]
        if value_here == 'no visitors':
            variables_dict[i] = value_here
        else:
            variables_dict[i] = float(value_here) 
    return variables_dict

no_visits_cbgs = params[params['cost'] == 'no visitors'].cbg.tolist()
good_cbgs = [i for i in cbgs if i not in no_visits_cbgs]

print('Cbgs without visits: ', len(no_visits_cbgs))
print('Cbgs suitable for calculation: ', len(good_cbgs))

def compute_row(row,p_0,p_1,p_2,p_3,p_4,p_5,p_6):
    up_first = (row['H_Area_of_store']**p_0)*((1+p_1)**row['I_Includes_parking_lot'])*(row['R_Percentage_of_Visits_by_brand']**p_2)
    up_second = (row['J_POI_count_where_store_is']**p_3)*(row['K_POI_diversity_where_store_is']**p_4)*(row['L_Demographic_similarity']**p_5)
    down = row['G_Distance_between_cbg_and_store']**p_6
    return up_first*up_second/down

def huff_pred_single_cbg_by_cbg(df, p_df, cbg):
    rate = get_rate(df, cbg)
    df_temp = df[df['A_cbg'] == cbg].drop(columns = ['A_cbg'])
    dict_params = get_params(p_df, cbg)
    p_0 = dict_params['H_Area_of_store']
    if p_0 == 'no visitors':
        for i in ['assumed_utility','assumed_visits']:
            df_temp[i] = ['no visitors in 2018'] * len(df_temp)
        df_temp['A_cbg'] = [str(cbg)] * len(df_temp)
        return df_temp
    p_1 = dict_params['I_Includes_parking_lot']
    p_2 = dict_params['R_Percentage_of_Visits_by_brand']
    p_3 = dict_params['J_POI_count_where_store_is']
    p_4 = dict_params['K_POI_diversity_where_store_is']
    p_5 = dict_params['L_Demographic_similarity']
    p_6 = dict_params['G_Distance_between_cbg_and_store']
    total_visits = df_temp['D_Number_of_Visits'].sum()
    df_temp['assumed_utility'] = df_temp.apply(lambda row: compute_row(row,p_0,p_1,p_2,p_3,p_4,p_5,p_6),axis = 1)
    all_assumed_utility = df_temp['assumed_utility'].sum()
    df_temp['assumed_utility'] = df_temp['assumed_utility'].apply(lambda x: x/all_assumed_utility)
    df_temp['assumed_visits'] = df_temp['assumed_utility'].apply(lambda x: x*total_visits)
    df_temp['A_cbg'] = [str(cbg)] * len(df_temp)
    return df_temp

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
N = len(cbgs_complete)//(size-1)
index_here = rank * N
if rank < size -1:
    cbgs = cbgs[index_here:index_here+N]
else:
    cbgs = cbgs[index_here:]

for cbg in cbgs:
    ans_here = huff_pred_single_cbg_by_cbg(table, params, cbg)
    ans_here.to_csv('sub_2019pred_2018params/2019pred_2018params_' + str(cbg) + '.csv', index = False)

'''
def compute_a_store(target_id):
    loss_info_num = {}
    for i,c in enumerate(good_cbgs):
        ans = huff_loss_single(table, params, c,target_id)
        loss_info_num[c] = ans
        print(i)
    return loss_info_num, dict(pd.DataFrame(loss_info_num).T.sum(axis=0))

target_ids_complete = target_ids.copy()

total_info_num_cbg = {}
total_info_num_store = {}
for i, t in enumerate(target_ids):
    ans_cbg = compute_a_store(t)[0]
    ans_store = compute_a_store(t)[1]
    total_info_num_cbg[t] = ans_cbg
    total_info_num_store[t] = ans_store
    if ':' in t:
        with open('/Volumes/圆滚滚/store_closing/'+str(year) + '_dicts/' + t + '.json', 'w') as fp:
            json.dump(ans_store, fp)
    else:
        with open('/Volumes/圆滚滚/store_closing/'+str(year) + '_dicts/' + t + '.json', 'w') as fp:
            json.dump(ans_store, fp)
    print(t, 'finished')

total_info_cbg_df = pd.DataFrame(total_info_num_cbg)

def new(element):
    #e = ast.literal_eval(element)
    e = element
    count = 0
    for k,v in e.items():
        if k in target_ids:
            count += v
    return count

total_info_cbg_df = total_info_cbg_df.applymap(lambda x: new(x))
total_info_cbg_df.to_csv('/Volumes/圆滚滚/store_closing/PSO_pred_visits_loss_by_cbg_' + str(year) + '.csv')

total_info_store_df = pd.DataFrame(total_info_num_store)
total_info_store_df.to_csv('/Volumes/圆滚滚/store_closing/PSO_pred_visits_loss_by_store_' + str(year) + '.csv')
'''

'''
import os
p = '/home/yilunx/Desktop/new_PSO_20210616/sub_2019pred_2018params'
import pandas as pd
files = os.listdir(p)
d = [pd.read_csv(f) for f in files]
df = pd.concat(d)
df.shape
df.to_csv('/home/yilunx/Desktop/new_PSO_20210616/pred_visits_2019pred_2018params.csv', index = False)
'''
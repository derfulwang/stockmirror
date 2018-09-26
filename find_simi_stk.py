#coding: utf-8

import glob
import datetime
from functools import partial

import pandas as pd
import numpy as np

def data_loader(file_path='D:/to_delete/data'):
    fs = glob.glob(file_path + '/*.csv')
    dfs = {}
    for f in fs:
        df = pd.read_csv(
            f, index_col=0,
            dtype={'code':np.dtype('U')}
        ).set_index('date')
        df.index = pd.to_datetime(df.index)
        code = df.code.head(1).values[0]
        dfs[code] = df
    return dfs
        
def person_dis(a, b):
    return np.corrcoef(np.vstack((a,b,)))[0,1]

def cos_dis(a, b):
    return np.dot(a,b) / np.sqrt(np.dot(a,a)*np.dot(b,b))


def find_most_similar(target, stock_pool, latest_date=None, win_size = 15, calc_col = 'close', before_days=10, top_num=20):
    '''
    traget: Series
    stock_poll: Dict, {code: DataFrame,...}, DataFrame的索引为DatetimeIndex
    '''
    if not latest_date:
        latest_date = datetime.date.today().strftime('%Y-%m-%d')  # 默认最近一天
    before_date = (datetime.date.today() - datetime.timedelta(days=1)).strftime('%Y-%m-%d') # 对比项的日期在此之前
    smilarity_res = {}
    target = target[:latest_date]
    for stk_code, stk in stock_pool.items():
        stk = stk[:before_date]
        smilarity_res[stk_code] = stk[calc_col].rolling(win_size).apply(
            partial(person_dis, target[calc_col].values[0:win_size]))
    tops = get_topk(smilarity_res, top_num=top_num)
    traget_code = target.code.values[0]
    most_similar = {traget_code: []}
    for code, dt, simi in tops:
        most_similar[traget_code].append({
            'code': code,
            'similarity': simi,
            'end_date': dt
        })
        #most_similar[traget_code].append((simi, stock_pool[code][:dt][-win_size:],))
    return most_similar


def get_topk(smilarities, top_num=10):
    topN = []
    for code, simi_series in smilarities.items():
        for dt, simi in simi_series.nlargest(top_num).items():
            topN.append((code, dt, simi,))
    topN.sort(key=lambda x:x[2], reverse=True)
    return topN[:top_num]

import dask
dfs = data_loader()
parallized_fms = dask.delayed(find_most_similar)
to_calc = ['000002','600362','002024','601601']
results = dask.compute(
    [parallized_fms(dfs[code], dfs) for code in to_calc],
    scheduler='multiprocessing')

dfs = []
for l in results[0]:
    for c,r in l.items():
        df = pd.DataFrame(r)
        df['target'] = c
        dfs.append(df)
df = pd.concat(dfs)
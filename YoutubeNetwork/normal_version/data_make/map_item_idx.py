# -*- coding: utf-8 -*-

import pickle
import pandas as pd

"""
读入的数据是，一个user id，对应的itemid，brandid，middleid和点击的时间
"""

PATH_TO_DATA = '/Data/sladesha/youtube/Neg_Data/click_brand_msort_data_20180415.txt'
data = pd.read_csv(PATH_TO_DATA, sep='\t', header=None)
data.columns = ['UId', 'ItemId', 'BrandId', 'MiddlesortId', 'ClickTime', 'Date']
data = data[['UId', 'ItemId', 'BrandId', 'MiddlesortId', 'ClickTime']]



"""
key就是所有的category的内容

m就是把这些key映射成为对应的id，从0开始

"""
def build_map(df, col_name):
    key = df[col_name].unique().tolist()
    m = dict(zip(key, range(len(key))))

    df[col_name] = df[col_name].map(lambda x: m[x])
    return m, key

"""
item_key是所有的item_id，原始的物品名称,
item_map是吧item_id转化为为对应的从0开始编号的index，是一个dictionary
经过下面的转换，所data里面所有的user和item_id都换成了相应的index
"""
item_map, item_key = build_map(data, 'ItemId')
brand_map, brand_key = build_map(data, 'BrandId')
msort_map, msort_key = build_map(data, 'MiddlesortId')
user_map, user_key = build_map(data, 'UId')

user_count, item_count, brand_count, msort_count, example_count = \
    len(user_key), len(item_key), len(brand_key), len(msort_key), len(data)
item_brand = data[['ItemId', 'BrandId']]
"""
item_brand保存的是吧item的inde和brand的index进行一个转换的表格

"""
item_brand = item_brand.drop_duplicates()
"""
这里去交集不知道是什么意思，这里lset的值就是我所有的item_id对应的index
"""
lset = set(item_map.values()) & set(item_brand['ItemId'].tolist())

"""
brand_list保存的是所有数据去重之后，所有的brand的新的index，
"""
brand_list = item_brand['BrandId'].tolist()

item_msort = data[['ItemId', 'MiddlesortId']]
item_msort = item_msort.drop_duplicates()
msort_list = item_msort['MiddlesortId'].tolist()

with open('remap.pkl', 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((item_key, brand_key, msort_key, user_key), f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(brand_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(msort_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, item_count, brand_count, msort_count, example_count), f, pickle.HIGHEST_PROTOCOL)

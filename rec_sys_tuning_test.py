import os
import pandas as pd
import numpy as np
from Rec_Sys.Data import Data
from Rec_Sys.Tuner import tunning

base = os.path.join(os.path.curdir, 'data')
path = os.path.join(base, 'light_2019-Oct.parquet')
df = pd.read_parquet(path, engine='fastparquet')

data = Data(df)

product_lookup = data.get_item_lookup()
print('prdouct_lookup 생성 완료')

no_cvs_view_ratio_matrix, no_cvs_view_ratio_data = data.get_view_ratio_df('no_conversion')
print('User_Item_Matrix 생성 완료, Data 생성 완료')

params = {
    'factors' : np.arange(100, 210, 10),
    'alpha' : np.arange(1, 41, 5),
    'iterations' : np.arange(20, 110, 10)
}

no_cvs_user_random = tunning(no_cvs_view_ratio_matrix, product_lookup, no_cvs_view_ratio_data, params, n_iters=3)
print(no_cvs_user_random)
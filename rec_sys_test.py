import pandas as pd
import os
from implicit.als import AlternatingLeastSquares
from Rec_Sys.Data import Data
from Rec_Sys.Model import ALS


base = os.path.join(os.path.curdir, 'data')
path = os.path.join(base, 'light_2019-Oct.parquet')
df = pd.read_parquet(path, engine='fastparquet')

data = Data(df)

product_lookup = data.get_item_lookup()
print('prdouct_lookup 생성 완료')

no_cvs_view_ratio_matrix, no_cvs_view_ratio_data = data.get_view_ratio_df('no_conversion')
print('User_Item_Matrix 생성 완료, Data 생성 완료')

als = AlternatingLeastSquares()
print('AlternatingLeastSquares 인스턴스 생성 완료')
base_model = ALS(als, no_cvs_view_ratio_matrix, product_lookup, no_cvs_view_ratio_data)
score = base_model.get_score()
print(score)
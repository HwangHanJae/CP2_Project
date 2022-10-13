import datetime
import numpy as np
import scipy.sparse as sparse
import pandas as pd

class Data():
  """
  입력 받은 df(light_2019-Oct.parquet)를
  No conversion Users, Conversion Users 집단으로 나누어주는 클래스
  """
  def __init__(self, df):
    self.df = df
    
  
  def _after_11_user_remove(self):
    """
    UTC 문자 제거 이후 4시간 뒤에 11월 이후 데이터를 제거하는 함수
    """
    copy_df = self.df.copy()
    copy_df['event_time'] = copy_df['event_time'].apply(lambda x : x[:-4]).astype('datetime64')
    copy_df['event_time'] = copy_df['event_time'] + datetime.timedelta(hours=4)
    copy_df = copy_df.loc[copy_df['event_time'] < '2019-11-01']

    return copy_df

  def _get_users(self, users_type = 'no_conversion'):
    """
    user_type에 따라
    전환이 일어나지 않은 유저, 전환이 일어난 유저를 파악하여
    파라미터에 맞는 데이터를 추출 후 반환

    Parameters
    ----------
    user_type : 유저의 타입, defalut = no_conversion
    """
    df = self._after_11_user_remove()
    data = df[['user_id','product_id','event_type']]
    if users_type == 'no_conversion':
      drop_user_id = data.loc[data['event_type'] != 'view', 'user_id']
      data = data.loc[~data['user_id'].isin(drop_user_id)].reset_index(drop=True)
      data['event_type'] = data['event_type'].astype('object')

      return data
    elif users_type == 'conversion':
      cart_user_id = set(data.loc[data['event_type'] == 'cart','user_id'].unique())
      purchase_user_id = set(data.loc[data['event_type'] == 'purchase', 'user_id'].unique())
      all_id = cart_user_id.union(purchase_user_id)
      data = data.loc[data['user_id'].isin(all_id)].reset_index(drop=True)
      data['event_type'] = data['event_type'].astype('object')

      return data

  def _get_view_count_grouped(self, users_type = 'no_conversion'):
    """
    view_count를 기반으로 가중치를 설정 후 파라미터에 맞는 그룹을 반환

    Parameters
    ----------
    user_type : 유저의 타입, defalut = no_conversion
    """
    data = self._get_users(users_type)
    if users_type == "no_conversion":
      grouped = data.groupby(['user_id','product_id'])['event_type'].count()
      grouped =grouped.reset_index()
      data = grouped.rename(columns = {'event_type' : 'view_count'})

      return data
    elif users_type =='conversion':
      grouped = data.groupby(['user_id','product_id','event_type'])['event_type'].count()
      grouped = pd.DataFrame(grouped).rename(columns={'event_type' : 'count'}).reset_index()

      table = grouped.pivot_table(index=['user_id','product_id'], columns=['event_type'], values=['count'])
      table = table.reset_index()
      table.columns = ['user_id','product_id','cart','purchase','view_count']
      table = table.fillna(0)

      table['cart'] = table['cart'].astype('int')
      table['purchase'] = table['purchase'].astype('int')
      table['view_count'] = table['view_count'].astype('int')

      table.loc[table['view_count'] == 0, 'view_count'] = 2

      data = table[['user_id','product_id','view_count']]

      return data

  def get_view_count_df(self, users_type = 'no_conversion'):
    """
    파라미터에 맞는 User-Item Matrix(Sparse Matrix), Data를 생성 후 반환

    Parameters
    ----------
    user_type : 유저의 타입, defalut = no_conversion
    """
    self.data = self._get_view_count_grouped(users_type)
    num_user = self.data['user_id'].nunique()
    num_item = self.data['product_id'].nunique()

    users = list(np.sort(self.data['user_id'].unique()))
    products = list(self.data['product_id'].unique())
    count = list(self.data['view_count'])

    rows = self.data['user_id'].astype('category').cat.codes
    self.data['user_id_code'] = self.data['user_id'].astype('category').cat.codes

    cols = self.data['product_id'].astype('category').cat.codes
    self.data['product_id_code'] = self.data['product_id'].astype('category').cat.codes

    user_item_matrix = sparse.csr_matrix((count, (rows, cols)), shape=(num_user, num_item))

    return user_item_matrix, self.data

  def _get_view_ratio_grouped(self, users_type  = "no_conversion"):
    """
    view_ratio를 기반으로 가중치를 설정 후 파라미터에 맞는 그룹을 반환

    Parameters
    ----------
    user_type : 유저의 타입, defalut = no_conversion
    """
    data = self._get_users(users_type)
    if users_type == 'no_conversion':
      grouped = data.groupby(['user_id','product_id'])['event_type'].count()
      grouped = grouped.reset_index()
      grouped = grouped.rename(columns = {'event_type' : 'view_count'})

      total_event_type = data.groupby(['user_id'])['product_id'].count()
      total_event_type = total_event_type.reset_index()
      total_event_type = total_event_type.rename(columns={'product_id' : 'total_view'})

      data = grouped.merge(total_event_type, on='user_id')
      data['view_ratio'] =  (data['view_count'] / data['total_view']) * 100

      return data
    elif users_type == 'conversion':
      grouped = data.groupby(['user_id','product_id','event_type'])['event_type'].count()
      grouped = pd.DataFrame(grouped).rename(columns={'event_type' : 'count'}).reset_index()

      table = grouped.pivot_table(index=['user_id','product_id'], columns=['event_type'], values=['count'])
      table = table.reset_index()
      table.columns = ['user_id','product_id','cart','purchase','view_count']
      table = table.fillna(0)

      table['cart'] = table['cart'].astype('int')
      table['purchase'] = table['purchase'].astype('int')
      table['view_count'] = table['view_count'].astype('int')

      table.loc[table['view_count'] == 0, 'view_count'] = 2

      temp = table[['user_id','product_id','view_count']]
      total_view = temp.groupby(['user_id'])['view_count'].sum()
      total_view = total_view.reset_index()
      total_view = total_view.rename(columns={'view_count' : 'total_view'})

      data = temp.merge(total_view, on='user_id')
      data['view_ratio'] = (data['view_count'] / data['total_view']) * 100

      data = data[['user_id','product_id','view_ratio']]
      return data

  def get_view_ratio_df(self, users_type = 'no_conversion'):
    """
    파라미터에 맞는 User-Item Matrix(Sparse Matrix), Data를 생성 후 반환

    Parameters
    ----------
    user_type : 유저의 타입, defalut = no_conversion
    """
    self.data = self._get_view_ratio_grouped(users_type)
    num_user = self.data['user_id'].nunique()
    num_item = self.data['product_id'].nunique()

    users = list(np.sort(self.data['user_id'].unique()))
    products = list(self.data['product_id'].unique())
    ratio = list(self.data['view_ratio'])

    rows = self.data['user_id'].astype('category').cat.codes
    self.data['user_id_code'] = self.data['user_id'].astype('category').cat.codes

    cols = self.data['product_id'].astype('category').cat.codes
    self.data['product_id_code'] = self.data['product_id'].astype('category').cat.codes

    user_item_matrix = sparse.csr_matrix((ratio, (rows, cols)), shape=(num_user, num_item))
    
    return user_item_matrix, self.data
  def get_item_lookup(self):
    """
    product의 정보가 담긴
    product_lookup 테이블을 생성 후 반환
    """
    copy_df = self._after_11_user_remove()
    product_lookup = copy_df[['product_id','category_code','brand']].drop_duplicates('product_id').reset_index(drop=True).sort_values('product_id')
    return product_lookup

from implicit.als import AlternatingLeastSquares
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

class ALS():
  """
  implicit 라이브러리를 이용하여 필요한 기능을 담았습니다.

  Parameters
  ----------
  model : implicit 라이브러리의 AlternatingLeastSquares() 클래스로 생성된 인스턴스
  user_item_matrix : 사용자가 정의한 User-Item Matrix(Sparse Matrix)
  item_lookup : item(product)에 대한 정보를 담은 테이블
  data : User-Item Matrix를 만든 원본데이터

  """
  def __init__(self, model, user_item_matrix, item_lookup, data):

    self.model = model
    self.metrics_model = model
    self.user_item_matrix = user_item_matrix
    self.item_lookup = item_lookup
    self.data = data

  def fit(self):
    """
    행렬분해의 ALS를 이용하여 모델을 학습합니다.

    Parameters
    ----------
    user_item_matrix : 사용자가 정의한 User-Item Matrix(Sparse Matrix)
    """
    user_item_matrix = self.user_item_matrix.copy()
    self.model.fit(user_item_matrix)
    
  def _user_id_2_code(self, user_id):
    """
    입력받은 user_id를 User_Item_Matrix에 있는 user_id_code로 바꾸어주는 함수

    Parameters
    ----------
    user_id : 유저 ID

    """
    user_id_code = self.data.loc[self.data['user_id'] == user_id, 'user_id_code'].unique()[0]
    return user_id_code

  def _product_id_2_code(self, product_id):
    """
    입력받은 product_id를 User-Item-Matrix에 있는 product_id_code로 바꾸어주는 함수

    Parameters
    ----------
    product_id : 상품 ID
    """
    product_id_code = self.data.loc[self.data['product_id'] == product_id, 'product_id_code'].unique()[0]
    return product_id_code

  def _code_2_product_id(self, product_id_code):
    """
    입력받은 product_id_code를 User-Item-Matrix에 있는 product_id로 바꾸어주는 함수

    Parameters
    ----------
    product_id_code : 상품 ID code    
    """
    product_id = self.data.loc[self.data['product_id_code'] == product_id_code, 'product_id'].unique()[0]
    return product_id

  def get_recom_product(self, user_id, n = 10):
    """  
    user_id에 맞는 product를 n개 만큼 추천하여 데이터프레임 형태로 반환하는 함수

    Parameters
    ----------
    user_id : 유저 ID
    n : 추천 받게 될 item의 수
    """ 

    # user_id_2_code 함수를 이용하여 유저의 ID를 user_id_code로 변환합니다
    user_id_code = self._user_id_2_code(user_id)
  
    # model의 recommend를 이용하여 추천받는 제품의 id를 추출합니다.
    # 이때 추천 받는 제품의 id는 product_id가 아니라 product_id_code 입니다.
    recommended = self.model.recommend(user_id_code, self.user_item_matrix[user_id_code], N=n)[0]
    #결과를 담을 리스트를 초기화 합니다.
    results = []
    # 추천 받은 id를 돌면서 item_lookup 테이블에서 해당 product의 정보를 찾아 결과에 담습니다.
    for product_id_code in recommended:
      
      recommended_product_id = self._code_2_product_id(product_id_code)
      result = self.item_lookup.loc[self.item_lookup['product_id'] == recommended_product_id]
      results.append(result)
      
    return pd.concat(results)
  
  def get_user_topN_product(self,user_id,column, n = 20):
    """
    유저가 특정 기준값이 높은 제품 N개를 반환

    Parameters
    ----------
    user_id : 유저 ID
    column : 값을 확인할 기준이 되는 컬럼
    n : 반환할 Item 수
    """
    #입력받은 user_id 를 기준으로 column이 높은 순으로 정렬하여 product_id를 추출
    product_ids = self.data.loc[self.data['user_id'] == user_id].sort_values(column, ascending=False)[:n]['product_id'].values
    #입력받은 user_id 를 기준으로 column이 높은 순으로 정렬하여 column을 추출
    product_values = self.data[self.data['user_id'] == user_id].sort_values(column, ascending=False)[:n][column].values

    results = []
    #item_lookup 테이블에서 id에 맞는 데이터프레임을 찾음
    for i in product_ids:
      result = self.item_lookup.loc[self.item_lookup['product_id'] == i]
      results.append(result)

    #결과를 확인하기 쉽게 데이터프레임으로 반환
    frame = pd.concat(results)
    frame[column] = product_values

    return frame

  def get_explain(self, user_id, item_id, column):
    """
    사용자에게 제품이 추천된 이유를 반환하는 함수

    Parameters
    ----------
    user_id : 유저 ID
    item_id : Item(product) ID
    column : 확인할 컬럼
    """
    #입력받은 user_id, item_id를 user_id_code, product_id_code로 바꾸어줌
    user_id_code = self._user_id_2_code(user_id)
    product_id_code = self._product_id_2_code(item_id)

    #implicit라이브러리의 explain 함수를 사용하여 결과값을 반환
    total_score, top_contributions, user_weights = self.model.to_cpu().explain(user_id_code, self.user_item_matrix, product_id_code)

    results = []
    categorys = []
    brands = []
    scores = []
    # id에 해당하는 user_id, product_id, column, category, brand를 찾기
    for id_, score_ in top_contributions:
      product_id = self._code_2_product_id(id_)
      result = self.data.loc[(self.data['product_id'] == product_id) & (self.data['user_id'] == user_id)][['user_id','product_id',column]]
      category = self.item_lookup.loc[self.item_lookup['product_id'] == product_id, 'category_code'].unique()[0]
      brand = self.item_lookup.loc[self.item_lookup['product_id'] == product_id, 'brand'].unique()[0]

      results.append(result)
      categorys.append(category)
      brands.append(brand)
      scores.append(score_)

    #결과를 확인하기 쉽게 데이터프레임으로 반환
    frame = pd.concat(results)
    frame['score'] = scores
    frame['category'] = categorys
    frame['brand'] = brands
    
    frame = frame[['user_id', 'product_id','category','brand',column, 'score']]
    return frame, total_score

  def _get_train_test(self,percentage=.2, seed=42):
    """
    score를 구하기 위하여 train, test 데이터를 나누어주는 함수
    파라미터로 들어오는 percentage만큼 train_set의 값을 0으로 만들어 줌
    test_set는 기존의 User-Item Matrix에서 0이 아닌값으로 모두 1로 만들어 줌

    Parameters
    ----------
    percentage : 감추고 싶은 데이터의 비율
    seed : random seed

    """
    #원본 데이터를 test_set, train_set에 복사
    test_set = self.user_item_matrix.copy()
    train_set = self.user_item_matrix.copy()

    #relevant(선호 혹은 평가)여부를 확인하기 위하여 test_set에서 0이 아닌값을 1로 만들어 줌 
    test_set[test_set != 0] = 1

    #train_set에서 0이 아닌 x축, y축을 추출
    nonzero_idxs = train_set.nonzero()
    #x, y를 짝을지어 저장
    nonzero_pairs = list(zip(nonzero_idxs[0], nonzero_idxs[1]))

    #랜덤 시드를 적용
    random.seed(seed)
    #주어진 비율로 샘플을 추출
    n_samples = int(np.ceil(percentage * len(nonzero_pairs)))
    samples = random.sample(nonzero_pairs, n_samples)

    
    user_idxs = [index[0] for index in samples]
    item_idxs = [index[1] for index in samples]

    #샘플에 해당 하는 값들을 평가한적이 없도록 보이기 위하여 0으로 감춤
    train_set[user_idxs, item_idxs] = 0
    
    train_set.eliminate_zeros()

    self.zero_user_idxs = user_idxs
    self.zero_item_idxs = item_idxs
    self.train_set = train_set
    self.test_set = test_set
    

  def get_score(self, percentage =.2, seed=42, k = 10, method='hit_at_k', n_samples = 10000, verbose=True):
    """
    train_set로 학습하고 test_set와 비교하여 method 파라미터를 이용하여 추천시스템의 성능을 평가하는 함수
    
    Parameters
    ----------
    percentage : 감추고 싶은 데이터의 비율, default = .2
    seed : random seed, default = 42
    k : 추천할 아이템의 수, default = 10
    method : 평가 지표 - hit_at_k, precision_at_k
    n_samples : 평가할 user의 수, default = 10000

    """
    # 입력받은 n_samples가 최대값이상 이면 최대값으로 적용
    max_num = self.data['user_id_code'].nunique()
    if max_num <= n_samples:
      n_samples = max_num

    # train, test로 데이터가 분리
    self._get_train_test(percentage, seed)
    # metrics_model이 학습
    self.metrics_model.fit(self.train_set)
    # 샘플 user를 랜덤으로 추출
    random_state = np.random.RandomState(seed)
    user_id_code_samples = random_state.choice(self.data['user_id_code'], n_samples)

    #method 방법에 따라 scores 값을 반환
    if method == 'hit_at_k':
      scores = self._hit_at_k(user_id_code_samples, k, verbose= verbose)
      return scores

    elif method =='precision_at_k':
      scores = self._precision_at_k(user_id_code_samples, k, verbose=verbose)
      return scores

  def _hit_at_k(self, user_id_code_samples, k, verbose =True):
    """
    k개의 추천 중 relevant한것(존재)이 있다면 1, 아니면 0을 반환하여 측정
    추천을 받은 user 수 만큼 나누어 평균을 반환

    Parameters
    ----------
    user_id_code_samples : 샘플링한 유저의 ID 리스트
    k : 추천할 아이템의 수
    """
    if verbose:
      scores = []
      #입력받은 user_id_code_samples를 돌면서
      for user_id_code_sample in tqdm(user_id_code_samples):
        #해당 유저에게 추천하는 아이템을 추출
        recommedation_ids = self.metrics_model.recommend(user_id_code_sample, self.train_set[user_id_code_sample], N=k)[0]

        results = []
        # 추천 받은 아이템이 유저가 선호(혹은 평가)했는지 확인
        for id_ in recommedation_ids:
          result = self.test_set[user_id_code_sample, id_]
          results.append(result)
        # 만약 결과 리스트안에 1이 있다면 1을 입력, 아니면 0을 입력
        if 1 in results:
          scores.append(1)
        else:
          scores.append(0)
          #결과를 평균내어 반환
      return np.mean(scores)
    else:
      scores = []
      #입력받은 user_id_code_samples를 돌면서
      for user_id_code_sample in user_id_code_samples:
        #해당 유저에게 추천하는 아이템을 추출
        recommedation_ids = self.metrics_model.recommend(user_id_code_sample, self.train_set[user_id_code_sample], N=k)[0]

        results = []
        # 추천 받은 아이템이 유저가 선호(혹은 평가)했는지 확인
        for id_ in recommedation_ids:
          result = self.test_set[user_id_code_sample, id_]
          results.append(result)
        # 만약 결과 리스트안에 1이 있다면 1을 입력, 아니면 0을 입력
        if 1 in results:
          scores.append(1)
        else:
          scores.append(0)
          #결과를 평균내어 반환
      return np.mean(scores)
  def _precision_at_k(self, user_id_code_samples, k, verbose=True):
    """
    k개의 추천 중 사용자가 relevant(선호 혹은 평가)한 아이템이 얼마나 존재하는지 측정
    추천을 받은 user 수 만큼 나누어 평균을 반환

    Parameters
    ----------
    user_id_code_samples : 샘플링한 유저의 ID 리스트
    k : 추천할 아이템의 수
    """
    if verbose:
      scores = []
      #입력받은 user_id_code_samples를 돌면서
      for user_id_code_sample in tqdm(user_id_code_samples):
        #해당 유저에게 추천하는 아이템을 추출
        recommedation_ids = self.metrics_model.recommend(user_id_code_sample, self.train_set[user_id_code_sample], N=k)[0]
        results = []
        # 추천 받은 아이템이 유저가 선호(혹은 평가)했는지 확인
        for id_ in recommedation_ids:
          result = self.test_set[user_id_code_sample, id_]
          results.append(result)
        # 유저가 추천받은 아이템들을 얼마나 선호(혹은 평가)했는지 추출
        scores.append(np.mean(results))
        # 결과를 평균내어 반환
      return np.mean(scores)
    else:
      scores = []
      #입력받은 user_id_code_samples를 돌면서
      for user_id_code_sample in user_id_code_samples:
        #해당 유저에게 추천하는 아이템을 추출
        recommedation_ids = self.metrics_model.recommend(user_id_code_sample, self.train_set[user_id_code_sample], N=k)[0]
        results = []
        # 추천 받은 아이템이 유저가 선호(혹은 평가)했는지 확인
        for id_ in recommedation_ids:
          result = self.test_set[user_id_code_sample, id_]
          results.append(result)
        # 유저가 추천받은 아이템들을 얼마나 선호(혹은 평가)했는지 추출
        scores.append(np.mean(results))
        # 결과를 평균내어 반환
      return np.mean(scores)

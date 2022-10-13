import numpy as np
from implicit.als import AlternatingLeastSquares
from Rec_Sys.Model import ALS
from tqdm import tqdm
import pandas as pd

def tunning(user_item_matrix, product_lookup, data, params, n_iters = 10, metrics = 'hit_at_k', verbose=False, tuner = 'random'):
  if tuner == 'random':
    results = []
    for _ in tqdm(range(n_iters)):
      factor = np.random.choice(params["factors"])
      alpha = np.random.choice(params['alpha'])
      iteration = np.random.choice(params['iterations'])
      als = AlternatingLeastSquares(factors = factor, alpha=alpha, iterations = iteration)
      model = ALS(als, user_item_matrix, product_lookup, data)
      
      score = model.get_score(method = metrics, verbose= verbose)
      result = [factor, alpha, iteration, score]
      results.append(result)
    frame = pd.DataFrame(results, columns = ['factor','alpha','iteration',metrics])
    frame = frame.sort_values(metrics, ascending=False)[:5]

    return frame
  elif tuner == 'grid':
    results = []
    for factor in tqdm(params['factors']):
      for alpha in params['alpha']:
        for iteration in params['iterations']:
          als = AlternatingLeastSquares(factors = factor, alpha=alpha, iterations = iteration)
          model = ALS(als, user_item_matrix, product_lookup, data)

          score = model.get_score(method = metrics, verbose=verbose)
          result = [factor, alpha, iteration, score]
          results.append(result)
    frame = pd.DataFrame(results, columns = ['factor','alpha','iteration',metrics])
    frame = frame.sort_values(metrics, ascending=False)[:5]

    return frame
"""
@module correlations
Tools for analyzong the parser behavior such as accuracy().
"""

import torch

from . import register_analysis_fn

@register_analysis_fn('pearson-r')
def correlation_pearson_r(inputs_a, inputs_b, inputs_pairinfo):
  """
  Calculates Pearson's r.
  
  @param inputs List of dictionaries. Refer to 'corpus.py'' for more details.
  
  @return Dictionary with key 'pearson_r'.
  """
  batch_size = len(inputs_pairinfo)

  golden = torch.zeros(batch_size)
  predict = torch.zeros(batch_size)
  for i, score in enumerate(inputs_pairinfo):
    golden[i] = score['sts']
    predict[i] = score['sts_predict']
  
  x_norm = golden - torch.sum(golden) / batch_size
  y_norm = predict - torch.sum(predict) / batch_size

  pearson_r = torch.sum(x_norm * y_norm) / torch.sqrt(torch.sum(x_norm * x_norm) * torch.sum(y_norm * y_norm))

  return {'pearson-r': round(pearson_r.item(), 4)}

@register_analysis_fn('mse')
def correlation_mse(inputs_a, inputs_b, inputs_pairinfo):
  """
  Calculates Mean Squared Error(MSE).
  
  @param inputs List of dictionaries. Refer to 'corpus.py' for more details.
  
  @return Dictionary with key 'mse'.
  """
  batch_size = len(inputs_pairinfo)

  golden = torch.zeros(batch_size)
  predict = torch.zeros(batch_size)
  for i, score in enumerate(inputs_pairinfo):
    golden[i] = score['sts']
    predict[i] = score['sts_predict']
  
  d = torch.pow(golden - predict, 2)

  mse = torch.sum(d) / batch_size

  return {'mse': round(mse.item(), 4)}
"""
@module run_sparse_dist

Implements sparse distribution running introduced within TreeLSTM(Kai, 2016).
"""

import torch
import torch.nn as nn

from . import register_run_fn

@register_run_fn('sparse-dist')
def run_sparse_dist(model, inputs_a, inputs_b, inputs_pairinfo, config):
  """
  Sparse distribution running implemented.

  @param model STS Model designed for spase distribution. Must have `r_size` attribute.
  @param inputs_a List of dictionaries. Refer to 'dataset.py' for more details.
  @param inputs_b List of dictionaries. Refer to 'dataset.py' for more details.
  @param inputs_pairinfo List of dictionaries.
  @param config Dictionary containing required configurations.
  """
  device = next(model.parameters()).device
  batch_size = len(inputs_a)

  scores = model.run(inputs_a, inputs_b)

  r_size = model.r_size

  # FIXME: assertion that score is 0~5 range
  MAX_SCORE = 5
  step = MAX_SCORE/(r_size-1)
  sparse_vector = torch.arange(0, MAX_SCORE+step, step=step, device=device).detach().unsqueeze(0)
  assert sparse_vector.size(1) == r_size

  scores = torch.sum(torch.exp(scores) * sparse_vector, dim=1)

  for i, score in enumerate(scores):
    inputs_pairinfo[i]['sts_predict'] = round(score.item(), 2)
  
  return inputs_pairinfo


"""
@module loss_sparse_dist

Implements sparse distribution loss introduced within TreeLSTM(Kai, 2016).
"""

import torch
import torch.nn as nn

from math import log

from . import register_loss_fn

@register_loss_fn('sparse-dist')
def loss_sparse_dist(model, inputs_a, inputs_b, inputs_pairinfo, **kwargs):
  """
  Sparse distribution loss implemented.

  @param model STS Model designed for spase distribution. Must have `r_size` attribute.
  @param inputs_a List of dictionaries. Refer to 'dataset.py' for more details.
  @param inputs_b List of dictionaries. Refer to 'dataset.py' for more details.
  @param inputs_pairinfo List of dictionaries.
  @param **kwargs Keyword arguments to pass to model.run()

  @return loss FloatTensor with a single element as a loss value for the batch.
  """
  device = next(model.parameters()).device
  batch_size = len(inputs_a)

  scores = model.run(inputs_a, inputs_b)

  r_size = model.r_size

  # Generate sparse distribution for input batch
  sparse = torch.zeros(batch_size, r_size, device=device).detach()
  # FIXME: assertion that score is 0~5 range
  MAX_SCORE = 5
  for i, score in enumerate(inputs_pairinfo):
    score = score['sts']
    if score == MAX_SCORE:
      sparse[i, -1] = 1
    else:
      floor = int(score / MAX_SCORE * (r_size-1)) 
      ceil = floor + 1
      weight = score / MAX_SCORE * (r_size-1) - floor
      if weight < 1e-4:
        # On the grid
        sparse[i, floor] = 1
      else:
        # Between the grid
        sparse[i, ceil] = weight
        sparse[i, floor] = 1 - weight

  # Calculate Kullback-Liebler Divergence loss for sparce distribution
  # Note that KLDivLoss does not support batch-first shaping.
  loss = nn.KLDivLoss(reduction='batchmean')(scores, sparse)

  return loss

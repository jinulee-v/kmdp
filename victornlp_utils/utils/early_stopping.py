"""
@module Implements early stopping.
"""

import logging
import torch

class EarlyStopping():
  """
  @brief Criterion-based early stopping.

  Implments basic early stopping and model saving.
  Can save the best checkpoint by the valiadation loss.

  Usage:
  early_stopper = EarlyStopping() # initialize
  ...
  if early_stopper(epoch, loss, model, '.../.../.../model.pt'): # tries to early stop
    # early stopping the training sequence
    exit()
  # continue training

  Notes:
  This class makes use of the logging library for logging states.
  """

  def __init__(self, patience=3, eps=0.0, maximize=False):
    """
    @brief initializer for the EarlyStopping class.

    Waits for patience epochs even the criterion is not improved by eps.

    @param patience Patience.
    @param eps Epsilon.
    @param maximize If True, tries to maximize the criterion instead of minimizing it. Default: False.
    """
    self.patience = patience
    self.eps = eps
    self.maximize = maximize

    assert eps >= 0

    self.best_epoch = 0
    if maximize:
      self.best = float('-inf')
    else:
      self.best = float('inf')
    self.waits = 0
    self.logger = logging.getLogger()
  
  def __call__(self, epoch, criterion, model, save_dir):
    """
    @brief __call__ impl for early stopping.

    @param epoch Current epoch number for logging.
    @param criterion Value to update.
    @param model Model to save.
    @param save_dir Directory to save model.

    @return True if patience ran out (i.e. needs to early stop). False otherwise.
    """
    self.logger.info('Early stopping...')

    if self.maximize:
      if criterion > self.best + self.eps:
        self.logger.info('Updating best epoch...')
        self.logger.info('- Current epoch: %f', criterion)
        self.logger/info('- Previous best epoch: %f (epoch: %d)', self.best, self.best_epoch)
        self.best_epoch = epoch
        self.best = criterion
        self.waits = 0

        torch.save(model.state_dict(), save_dir)
      else:
        self.logger.info('Not updating...')
        self.logger.info('- Current epoch: %f', criterion)
        self.logger/info('- Previous best epoch: %f (epoch: %d)', self.best, self.best_epoch)

        self.waits += 1
        self.logger.info('Patience %d / %d', self.waits, self.patience)
        if self.waits >= self.patience:
          return True
    else:
      if criterion < self.best - self.eps:
        self.logger.info('Updating best epoch...')
        self.logger.info('- Current epoch: %f', criterion)
        self.logger.info('- Previous best epoch: %f (epoch: %d)', self.best, self.best_epoch)
        self.best_epoch = epoch
        self.best = criterion
        self.waits = 0

        torch.save(model.state_dict(), save_dir)
      else:
        self.logger.info('Not updating...')
        self.logger.info('- Current epoch: %f', criterion)
        self.logger.info('- Previous best epoch: %f (epoch: %d)', self.best, self.best_epoch)

        self.waits += 1
        self.logger.info('Patience %d / %d', self.waits, self.patience)
        if self.waits >= self.patience:
          return True
    
    return False

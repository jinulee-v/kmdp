"""
@module train

Script for training the sts analysis model.
"""

import os, sys
import json
from tqdm import tqdm
import logging
from datetime import datetime
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import *

from .victornlp_utils.corpora.dataset import *

from .victornlp_utils.embedding import embeddings

from .victornlp_utils.utils.early_stopping import EarlyStopping

from .model.model import sts_model
from .model.loss import sts_loss_fn
from .model.run import sts_run_fn
from .tools.analyze import sts_analysis_fn

def argparse_cmd_args() :
  parser = argparse.ArgumentParser(description='Train the sts analysis model.')
  parser.add_argument('--config-file', type=str, default='victornlp_sts/config.json')
  parser.add_argument('--title', type=str)
  parser.add_argument('--language', type=str, help='language. Currently supported: {korean, english}')
  parser.add_argument('--model', choices=sts_model.keys(), help='sts model. Choose sts name from default config file.')
  parser.add_argument('--loss-fn', choices=sts_loss_fn.keys(), help='sts loss function. Choose sts loss function from default config file.')
  parser.add_argument('--embedding', '-e', type=str, choices=embeddings.keys(), nargs='?', action='append', help='Embedding class names.')
  parser.add_argument('--epoch', type=int, help='training epochs')
  parser.add_argument('--batch-size', type=int, help='batch size for training')
  parser.add_argument('--optimizer', type=str, help='optimizer. Choose class name from torch.optim')
  parser.add_argument('--learning-rate', '--lr', type=float, help='learning rate')
  parser.add_argument('--device', type=str, help='device. Follows the torch.device format')
  parser.add_argument('--finetune-model-dir', type=str, help='Directory to model.pt & config.json subject to finetuning')
  
  args = parser.parse_args()
  
  return args

def main():
  """
  Training routine.
  """
  args = argparse_cmd_args()

  # Load configuration file
  config = None
  config_path = args.config_file
  if args.finetune_model_dir and args.config_file == 'victornlp_sts/config.json':
    config_path = args.finetune_model_dir + '/config.json'
  with open(config_path) as config_file:
    config = json.load(config_file)
  assert config
  
  train_config = config['train'] if 'train' in config else {}
  for arg, value in vars(args).items():
    if getattr(args, arg):
      train_config[arg] = value
  language_config = config['language'][train_config['language']]

  # Command line arguments override basic language configurations
  embeddings_list = train_config['embedding'] if 'embedding' in train_config else None
  if not embeddings_list:
    embeddings_list = language_config['embedding']

  embedding_config = {name:conf for name, conf in config['embedding'].items() if name in embeddings_list}
  if args.finetune_model_dir:
    for conf in embedding_config.values():
      if 'train' in conf:
        conf['train'] = 1
  model_config = config['model'][train_config['model']]

  # Set frequent constant variables
  language = train_config['language']
  model_name = train_config['model']

  now = datetime.now().strftime(u'%Y%m%d %H:%M:%S')
  title = train_config['title'] if 'title' in train_config else now + ' ' + language + ' ' + model_name
  if args.finetune_model_dir:
    title = title + ' fine-tuning'
  train_config['title'] = title

  # Extract only required features for clarity
  config = {
    'language': {
      language: language_config
    },
    'embedding': embedding_config,
    'model': {
      model_name: model_config
    },
    'train': train_config
  }

  # Directory for logging, config & model storage
  os.makedirs('models/' + title)

  formatter = logging.Formatter(fmt="%(asctime)s %(message)s")

  fileHandler = logging.FileHandler(filename='models/{}/train_{}.log'.format(title, now), encoding='utf-8')
  fileHandler.setFormatter(formatter)
  streamHandler = logging.StreamHandler()
  streamHandler.setFormatter(formatter)
  logger = logging.getLogger()
  logger.addHandler(fileHandler)
  logger.addHandler(streamHandler)
  logger.setLevel(logging.INFO)

  logger.info(title)

  logger.info('\n' + json.dumps(config, indent=4))
  with open('models/' + title + '/config.json', 'w', encoding='UTF-8') as f:
    json.dump(config, f, indent=4)
  
  # Load corpus
  logger.info('Preparing data...')
  train_path = language_config['corpus']['train']
  dev_path = language_config['corpus']['dev'] if 'dev' in language_config['corpus'] else None
  test_path = language_config['corpus']['test']
  labels_path = language_config['corpus']['labels']
  train_dev_ratio = language_config['corpus']['train_dev_ratio'] if 'train_dev_ratio' in language_config['corpus'] else None
  preprocessors = [dataset_preprocessors[alias] for alias in language_config['preprocessors']]
  
  train_dataset, dev_dataset, test_dataset, type_label = None, None, None, None
  with open(train_path['a']) as train_corpus_file_a, \
       open(train_path['b']) as train_corpus_file_b, \
       open(train_path['pair-info']) as train_corpus_file_pairinfo:
    train_dataset = VictorNLPPairDataset(json.load(train_corpus_file_a), json.load(train_corpus_file_b), json.load(train_corpus_file_pairinfo), preprocessors)
  with open(test_path['a']) as test_corpus_file_a, \
       open(test_path['b']) as test_corpus_file_b, \
       open(test_path['pair-info']) as test_corpus_file_pairinfo:
    test_dataset = VictorNLPPairDataset(json.load(test_corpus_file_a), json.load(test_corpus_file_b), json.load(test_corpus_file_pairinfo), preprocessors)

  # Split dev datasets
  if dev_path:
    with open(dev_path['a']) as dev_corpus_file_a, \
         open(dev_path['b']) as dev_corpus_file_b, \
         open(dev_path['pair-info']) as dev_corpus_file_pairinfo:
      dev_dataset = VictorNLPPairDataset(json.load(dev_corpus_file_a), json.load(dev_corpus_file_b), json.load(dev_corpus_file_pairinfo), preprocessors)
  else:
    if train_dev_ratio and train_dev_ratio < 1.:
      split = random_split(train_dataset, [int(len(train_dataset) * train_dev_ratio), len(train_dataset) - int(len(train_dataset)*train_dev_ratio)])
      train_dataset = split[0]
      dev_dataset = split[1]
    else:
      dev_dataset = VictorNLPDataset({})
  
  # Prepare DataLoader instances
  train_loader = DataLoader(train_dataset, train_config['batch_size'], shuffle=True, collate_fn=VictorNLPPairDataset.collate_fn)
  if dev_dataset:
    dev_loader = DataLoader(dev_dataset, train_config['batch_size'], shuffle=False, collate_fn=VictorNLPPairDataset.collate_fn)
  test_loader = DataLoader(test_dataset, train_config['batch_size'], shuffle=False, collate_fn=VictorNLPPairDataset.collate_fn)
  logger.info('done\n')
  
  # Create model
  logger.info('Preparing models and optimizers...')
  device = torch.device(train_config['device'])
  embedding_objs = [embeddings[embedding_type](embedding_config[embedding_type]).to(device) for embedding_type in embeddings_list]
  model = sts_model[model_name](embedding_objs, type_label, model_config)
  if args.finetune_model_dir:
    model.load_state_dict(torch.load(args.finetune_model_dir+ '/model.pt'))
  model = model.to(device)
  
  # Backpropagation settings
  optimizers = {
    'adam': Adam,
    'adagrad': Adagrad
  }
  loss_fn = sts_loss_fn[train_config['loss_fn']]
  optimizer = optimizers[train_config['optimizer']](model.parameters(), train_config['learning_rate'])
  run_fn = sts_run_fn[train_config['run_fn']]
  correlation = sts_analysis_fn['pearson-r']

  # Early Stopping settings
  if dev_dataset:
    es_config = train_config['early_stopping']
    early_stopper = EarlyStopping(es_config['patience'], es_config['eps'], es_config['maximize'])
  logger.info('done\n')
  
  # Training
  for epoch in range(1, train_config['epoch']+1):
    logger.info('-'*40)
    logger.info('Epoch: %d', epoch)
    
    logger.info('')
    logger.info('Train')
    model.train()
    
    iter = tqdm(train_loader)
    for i, batch in enumerate(iter):
      optimizer.zero_grad()
      loss = loss_fn(model, *batch)
      loss.backward()
      optimizer.step()
    
    # Validation
    if dev_dataset:
      logger.info('')
      logger.info('Validation')
      
      with torch.no_grad():
        model.eval()
        loss = 0
        cnt = 0
        for batch in tqdm(dev_loader):
          cnt += len(batch[0])
          loss += float(loss_fn(model, *batch)) * len(batch[0])
        logger.info('Dev loss: %f', loss/cnt)
        if early_stopper(epoch, loss/cnt, model, 'models/' + title + '/model.pt'):
          break
    
    # Accuracy(Correlation)
    logger.info('')
    logger.info('Correlation')
    
    with torch.no_grad():
      model.eval()
      for batch in tqdm(test_loader):
        # Call by reference modifies the original batch
        run_fn(model, *batch, language_config['run']) 
      
      logger.info(correlation(test_dataset._data_a, test_dataset._data_b, test_dataset._data_pairinfo))
      logger.info('-'*40)
      logger.info('')
  
  logger.info('Training completed.')
  logger.info('Check {} for logs, configurations, and the trained model file.'.format('models/' + title))

  
if __name__ == '__main__':
  main()

"""
@module train

Script for training the re analysis model.
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

from .model.model import re_model
from .model.loss import re_loss_fn
from .model.run import re_run_fn
from .tools.analyze import re_analysis_fn

from .kmdp.kmdp_utils import generate_kmdp_lengths_mask

def argparse_cmd_args() :
  parser = argparse.ArgumentParser(description='Train the re analysis model.')
  parser.add_argument('--config-file', type=str, default='victornlp_re/config.json')
  parser.add_argument('--title', type=str)
  parser.add_argument('--language', type=str, help='language. Currently supported: {korean, english}')
  parser.add_argument('--model', choices=re_model.keys(), help='RE model. Choose model name from default config file.')
  parser.add_argument('--loss-fn', choices=re_loss_fn.keys(), help='RE loss function.')
  parser.add_argument('--run-fn', choices=re_run_fn.keys(), help='RE run function.')
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
  if args.finetune_model_dir and args.config_file == 'victornlp_re/config.json':
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
  with open(train_path) as train_corpus_file:
    train_dataset = VictorNLPDataset(json.load(train_corpus_file), preprocessors)
  with open(test_path) as test_corpus_file:
    test_dataset = VictorNLPDataset(json.load(test_corpus_file), preprocessors)
  with open(labels_path) as type_label_file:
    type_label = json.load(type_label_file)

  # Split dev datasets
  if dev_path:
    with open(dev_path) as dev_corpus_file:
      dev_dataset = VictorNLPDataset(json.load(dev_corpus_file), preprocessors)
  else:
    if train_dev_ratio and train_dev_ratio < 1.:
      split = random_split(train_dataset, [int(len(train_dataset) * train_dev_ratio), len(train_dataset) - int(len(train_dataset)*train_dev_ratio)])
      train_dataset = split[0]
      dev_dataset = split[1]
    else:
      dev_dataset = VictorNLPDataset({})
  
  # Prepare DataLoader instances
  train_loader = DataLoader(train_dataset, train_config['batch_size'], shuffle=True, collate_fn=VictorNLPDataset.collate_fn)
  if dev_dataset:
    dev_loader = DataLoader(dev_dataset, train_config['batch_size'], shuffle=False, collate_fn=VictorNLPDataset.collate_fn)
  test_loader = DataLoader(test_dataset, train_config['batch_size'], shuffle=False, collate_fn=VictorNLPDataset.collate_fn)
  logger.info('done\n')
  
  # Create model
  logger.info('Preparing models and optimizers...')
  device = torch.device(train_config['device'])
  embedding_objs = [embeddings[embedding_type](embedding_config[embedding_type]).to(device) for embedding_type in embeddings_list]
  model = re_model[model_name](embedding_objs, type_label, model_config)
  if args.finetune_model_dir:
    model.load_state_dict(torch.load(args.finetune_model_dir+ '/model.pt'))
  model = model.to(device)
  
  # Backpropagation settings
  optimizers = {
    'adam': Adam
  }
  loss_fn = re_loss_fn[train_config['loss_fn']]
  optimizer = optimizers[train_config['optimizer']](model.parameters(), train_config['learning_rate'])
  run_fn = re_run_fn[train_config['run_fn']]
  accuracy = re_analysis_fn[train_config['analysis_fn']]

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
      lengths, mask = generate_kmdp_lengths_mask(batch, device)
      optimizer.zero_grad()
      loss = loss_fn(model, batch, lengths=lengths, mask=mask)
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
          lengths, mask = generate_kmdp_lengths_mask(batch, device)
          cnt += len(batch)
          loss += float(loss_fn(model, batch, lengths=lengths, mask=mask)) * len(batch)
        logger.info('Dev loss: %f', loss/cnt)
        if early_stopper(epoch, loss/cnt, model, 'models/' + title + '/model.pt'):
          break
    
    # Accuracy
    logger.info('')
    logger.info('Accuracy')
    
    with torch.no_grad():
      model.eval()
      for batch in tqdm(test_loader):
        # Call by reference modifies the original batch
        lengths, mask = generate_kmdp_lengths_mask(batch, device)
        run_fn(model, batch, language_config['run'], lengths=lengths, mask=mask) 
      
      logger.info(accuracy(test_dataset))
      logger.info('-'*40)
      logger.info('')
  
  logger.info('Training completed.')
  logger.info('Check {} for logs, configurations, and the trained model file.'.format('models/' + title))

  
if __name__ == '__main__':
  main()

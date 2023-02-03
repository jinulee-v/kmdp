"""
@module test

Supports evaluation, interactive&batched testing

Input data file must strictly follow VictorNLP format.

if 'pos' in input.keys():
  # Golden PoS tag information is given
  # (its format may vary among languages.)
else:
  # Perform language-specific PoS tagging(defined in victornlp_utils.pos_tagger)

if 'sts' in input.keys():
  # Golden sts label is given
  # Perform evaluation/analysis (specified by -a, --analyze)

Exceptionally, for stdin inputs, it only requires raw texts, a sentence per line.
PoS tags are automatically generated and no evaluation is performed.
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
from .victornlp_utils.pos_tagger import pos_taggers

from .victornlp_utils.utils.early_stopping import EarlyStopping

from .model.model import sts_model
from .model.loss import sts_loss_fn
from .model.run import sts_run_fn
from .tools.analyze import sts_analysis_fn

def parse_cmd_arguments():
  parser = argparse.ArgumentParser(description="Evaluate a model or analyze raw texts.")
  parser.add_argument('model_dir', type=str, help='Model directory that contains model.pt & config.json.')
  parser.add_argument('--data-file', type=str, help='File that contains VictorNLP format data. default: stdin(only raw texts)')
  parser.add_argument('-a', '--analyze', type=str, action='append', choices=sts_analysis_fn)
  parser.add_argument('--save-result', help='Print VictorNLP-format results to a file.')

  args = parser.parse_args()
  return args

def main():
  args = parse_cmd_arguments()

  # Load configuration file
  config = None
  config_path = args.model_dir + '/config.json'
  with open(config_path) as config_file:
    config = json.load(config_file)
  assert config
  
  train_config = config['train']
  language_config = config['language'][train_config['language']]
  embedding_config = config['embedding']
  model_config = config['model'][train_config['model']]
  
  # Set frequent constant variables
  language = train_config['language']
  model_name = train_config['model']
  now = datetime.now().strftime(u'%Y%m%d %H:%M:%S')
  title = train_config['title'] if 'title' in train_config else now + ' ' + language + ' ' + model_model
  train_config['title'] = title

  # Set logger
  file_formatter = logging.Formatter(fmt="%(message)s")
  stream_formatter = logging.Formatter(fmt="%(asctime)s %(message)s")

  fileHandler = logging.FileHandler(filename=args.model_dir + '/parse_{}.log'.format(now), encoding='utf-8')
  fileHandler.setFormatter(file_formatter)
  streamHandler = logging.StreamHandler()
  streamHandler.setFormatter(stream_formatter)
  logger = logging.getLogger()
  logger.addHandler(fileHandler)
  logger.addHandler(streamHandler)
  logger.setLevel(logging.INFO)

  logger.info(title)

  # Prepare data
  logger.info('Preparing data...')

  # Prepare evaluation data if file is given
  from_file = bool(args.data_file)
  args.data_file = language_config['corpus']['test']
  # pos_tagger = pos_taggers[language]
  assert language_config['preprocessors'][0] == 'word-count'
  preprocessors = [dataset_preprocessors[alias] for alias in language_config['preprocessors']]
  # preprocessors.insert(1, pos_tagger)

  if from_file:
    with open(args.data_file['a']) as inputs_a, \
       open(args.data_file['b']) as inputs_b, \
       open(args.data_file['pair-info']) as inputs_pairinfo:
      dataset = VictorNLPPairDataset(json.load(inputs_a), json.load(inputs_b), json.load(inputs_pairinfo), preprocessors)
  
      # Prepare DataLoader instances
      loader = DataLoader(dataset, train_config['batch_size'], shuffle=True, collate_fn=VictorNLPPairDataset.collate_fn)
      if args.analyze:
        # If evaluation mode, input must contain gold sts information
        for i in range(len(dataset)):
          assert 'sts' in dataset[i][2]
  else:
    logger.info('Receiving data from stdin...')
  
  # Create model
  logger.info('Preparing models...')
  device = torch.device(train_config['device'])
  embeddings_list = language_config['embedding']
  embedding_objs = [embeddings[embedding_type](embedding_config[embedding_type]).to(device) for embedding_type in embeddings_list]
  model = sts_model[model_name](embedding_objs, None, model_config)
  model.load_state_dict(torch.load(args.model_dir + '/model.pt'))
  model = model.to(device)

  run_fn = sts_run_fn[train_config['run_fn']]

  # Evaluation
  with torch.no_grad():
    if from_file:
      # From file data

      # Run and log time
      before = datetime.now()
      logger.info('Started at...' + before.strftime(u'%Y%m%d %H:%M:%S'))
      for batch in tqdm(loader):
        # Call by reference modifies the original batch
        run_fn(model, *batch, language_config['run'])
      after = datetime.now()
      logger.info('Finished at...' + after.strftime(u'%Y%m%d %H:%M:%S'))
      seconds = (after - before).total_seconds()
      logger.info('Total time: %.2fs (%.2f sents/s)', seconds, len(dataset)/seconds)
      logger.info('')

      # Run analysis functions
      if args.analyze:
        analyzers = {name:sts_analysis_fn[name] for name in args.analyze}
        for name, analyzer in analyzers.items():
          result = analyzer(dataset._data_a, dataset._data_b, dataset._data_pairinfo)
          logger.info('-'*40)
          logger.info(name)
          if isinstance(result, dict):
            # Dictionary results
            for key, value in result.items():
              logger.info('  {}: {}'.format(key, value))
          else:
            # Text results(TSV, pd.dataframe, ...)
            logger.info('\n' + str(result))
          logger.info('-'*40)
          logger.info('')

      # Save result if needed
      if args.save_result:
        with open(args.model_dir + 'parse_result_{}.log'.format(now), 'w') as out_file:
          json.dump(inputs, out_file, indent=4)

    else:
      # From stdin
      while True:
        sentence = input()
        # Conversion to VictorNLP format
        sentence = [{'text': sentence}]
        for preprocess in preprocessors:
          sentence = preprocess(sentence)

        # Run and log time
        before = datetime.now()
        logger.info('Started at...' + before.strftime(u'%Y%m%d %H:%M:%S'))
        parse_fn(model, sentence, language_config['run'])
        after = datetime.now()
        logger.info('Finisheded at...' + after.strftime(u'%Y%m%d %H:%M:%S'))
        seconds = (after - before).total_seconds()
        logger.info('Total time: %.2fs (%.2f sents/s)', seconds, len(dataset)/seconds)
        logger.info('')
        
        # Format and print result
        sentence = sentence[0]
        # TODO

if __name__ == '__main__':
  main()
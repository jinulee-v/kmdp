"""
@module parse

Supports evaluation, interactive&batched parsing.

Input data file must strictly follow VictorNLP format.

if 'pos' in input.keys():
  # Golden PoS tag information is given
  # (its format may vary among languages.)
else:
  # Perform language-specific PoS tagging(defined in victornlp_utils.pos_tagger)

if 'dependency' in input.keys():
  # Golden dependency tree is given
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
# from .victornlp_utils.pos_tagger import pos_taggers

from .victornlp_utils.utils.early_stopping import EarlyStopping

from .model.model import dp_model
from .model.loss import dp_loss_fn
from .model.parse import dp_parse_fn
from .tools.analyze import dp_analysis_fn

from .kmdp.kmdp_utils import *

def parse_cmd_arguments():
  parser = argparse.ArgumentParser(description="Evaluate a model or parse raw texts.")
  parser.add_argument('model_dir', type=str, help='Model directory that contains model.pt & config.json.')
  parser.add_argument('--data-file', type=str, help='File that contains VictorNLP format data. default: stdin(only raw texts)')
  parser.add_argument('-a', '--analyze', type=str, action='append', choices=dp_analysis_fn)
  parser.add_argument('--save-result', type=str, help='Print VictorNLP-format results to a file.')

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
  # Command line arguments override basic language configurations
  embeddings_list = train_config['embedding']
  embedding_config = config['embedding']
  parser_config = config['parser'][train_config['model']]
  
  # Set frequent constant variables
  language = train_config['language']
  parser_model = train_config['model']
  now = datetime.now().strftime(u'%Y%m%d %H:%M:%S')
  title = train_config['title'] if 'title' in train_config else now + ' ' + language + ' ' + parser_model
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

  # Load label data
  labels_path = language_config['corpus']['labels']
  with open(labels_path) as type_label_file:
    type_label = json.load(type_label_file)['dp_labels']

  # Prepare evaluation data if file is given
  from_file = bool(args.data_file)
  # pos_tagger = pos_taggers[language]
  assert language_config['preprocessors'][0] == 'word-count'
  preprocessors = [dataset_preprocessors[alias] for alias in language_config['preprocessors']]
  # preprocessors.insert(1, pos_tagger)

  if from_file:
    with open(args.data_file, 'r') as data_file:
      dataset = VictorNLPDataset(json.load(data_file), preprocessors)
      # Prepare DataLoader instances
      loader = DataLoader(dataset, train_config['batch_size'], shuffle=True, collate_fn=VictorNLPDataset.collate_fn)
      if args.analyze:
        # If evaluation mode, input must contain gold dependency tags.
        for i in range(len(dataset)):
          assert 'dependency' in dataset[i]
  else:
    logger.info('Receiving data from stdin...')
  
  # Create parser module
  logger.info('Preparing models...')
  device = torch.device(train_config['device'])
  embedding_objs = [embeddings[embedding_type](embedding_config[embedding_type]).to(device) for embedding_type in embeddings_list]
  parser = dp_model[parser_model](embedding_objs, type_label, parser_config)
  parser.load_state_dict(torch.load(args.model_dir + '/model.pt'))
  parser = parser.to(device)

  parse_fn = dp_parse_fn['greedy']

  # Evaluation
  with torch.no_grad():
    if from_file:
      # From file data

      # Parse and log time
      before = datetime.now()
      logger.info('Started at...' + before.strftime(u'%Y%m%d %H:%M:%S'))
      for batch in tqdm(loader):
        lengths, mask = generate_kmdp_lengths_mask(batch, device)
        # Call by reference modifies the original batch
        parse_fn(parser, batch, language_config['parse'], lengths=lengths, mask=mask)
      after = datetime.now()
      logger.info('Finished at...' + after.strftime(u'%Y%m%d %H:%M:%S'))
      seconds = (after - before).total_seconds()
      logger.info('Total time: %.2fs (%.2f sents/s)', seconds, len(dataset)/seconds)
      logger.info('')

      # Run analysis functions
      if args.analyze:
        analyzers = {name:dp_analysis_fn[name] for name in args.analyze}
        for name, analyzer in analyzers.items():
          result = analyzer(dataset)
          logger.info('-'*40)
          logger.info(name)
          if isinstance(result, dict):
            # Dictionary results
            for key, value in result.items():
              logger.info('{}\t{}'.format(key, value))
          else:
            # Text results(TSV, pd.dataframe, ...)
            logger.info('\n' + str(result))
          logger.info('-'*40)
          logger.info('')

      # Save result if needed
      if args.save_result:
        with open(args.save_result if args.save_result else args.model_dir + 'parse_result_{}.json'.format(now), 'w') as out_file:
          json.dump(dataset._data, out_file, indent=4, ensure_ascii=False)

    else:
      # From stdin
      while True:
        sentence = input()
        # Conversion to VictorNLP format
        sentence = [{'text': sentence}]
        for preprocess in preprocessors:
          sentence = preprocess(sentence)

        # Parse and log time
        before = datetime.now()
        logger.info('Started at...' + before.strftime(u'%Y%m%d %H:%M:%S'))
        parse_fn(parser, sentence, language_config['parse'])
        after = datetime.now()
        logger.info('Finisheded at...' + after.strftime(u'%Y%m%d %H:%M:%S'))
        seconds = (after - before).total_seconds()
        logger.info('Total time: %.2fs (%.2f sents/s)', seconds, len(dataset)/seconds)
        logger.info('')
        
        # Format and print result
        sentence = sentence[0]
        logger.info('; ' + sentence['text'])
        logger.info('# ' + ' '.join(
          ['+'.join(
            [morph['text'] + '/' + morph['pos_tag']
            for morph in wordphr])
          for wordphr in sentence['pos']]
        ))
        wordphrs = ['[ROOT]'] + sentence['text'].split()
        for arc in sentence['dependency']:
          logger.info('{}\t{}\t{}\t{}\t-> {}'.format(arc['dep'], arc['head'], arc['label'], wordphrs[arc['dep']], wordphrs[arc['head']]))
        logger.info('')

if __name__ == '__main__':
  main()
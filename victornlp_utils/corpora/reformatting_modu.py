"""
@module reformatting_modu
Convert Modu corpus to VictorNLP corpus format.
Is designed to reformat the file distributed in:
> https://corpus.korean.go.kr/
"""

import argparse6
import os, sys
import json
from torch.utils.data import random_split

from ..victornlp_utils.pos_tagger.pos_tagger import *

def modu_to_victornlp(modu_dp_file, modu_pos_file, modu_ner_file, train_file, dev_file, test_file, labels_file):
  victornlp = {}
  
  # process DP
  modu = json.load(modu_dp_file)
  dp_labels = set()
  for doc in modu['document']:
    for sent in doc['sentence']:
      id = sent.pop('id', None)
      sent['text'] = sent['form']
      sent.pop('form', None)
      sent['word_count'] = len(sent['text'].split())
      sent['dependency'] = sent['DP']
      sent.pop('DP', None)
      for arc in sent['dependency']:
        arc['dep'] = arc['word_id']
        arc.pop('word_id', None)
        arc.pop('word_form', None)
        arc.pop('dependent', None)
        if arc['head'] == -1:
          arc['head'] = 0
        arc['head'] = arc.pop('head')
        arc['label'] = arc.pop('label')
        if arc['label'] not in dp_labels:
          dp_labels.append(arc['label'])
      victornlp[id] = (sent)
  del modu
  dp_labels = sorted(list(dp_labels))

  # process PoS
  modu = json.load(modu_pos_file)
  pos_labels = set()
  for doc in modu['document']:
    for sent in doc['sentence']:
      id = sent.pop('id', None)
      target = victornlp[id]
      target['pos'] = []
      for morph in sent['morpheme']:
        while len(target['pos']) < morph['word_id']:
          target['pos'].append([])
        target['pos'][morph['word_id'] - 1].append({
          'id': morph['id'],
          'text': morph['form'],
          'pos_tag': morph['label'],
          'begin': morph['begin'],
          'end': morph['end']      # begin & end: temporay info for NER alignment
        })
        pos_labels.add(morph['label'])
  del modu
  pos_labels = sorted(list(pos_labels))
 
  # process NER
  modu = json.load(modu_ner_file)
  ner_labels = set()
  for doc in modu['document']:
    for sent in doc['sentence']:
      id = sent.pop('id', None)
      target = victornlp[id]
      target['named_entity'] = []

      i_wp = -1
      i_morph = 0
      def next_morph():
        i_morph += 1
        if i_wp < 0 or i_morph == len(target['pos'][i_wp]):
          i_wp += 1
          i_morph = 0
        return target['pos'][i_wp][i_morph]

      # re-write start/end as morpheme IDs, not character.
      for ne in sent['NE']:
        ne.pop('id', None)
        while next_morph()['begin'] < ne['begin']:
          pass
        start = next_morph()
        ne['begin'] = start['id']
        if start['end'] == ne['end']:
          end = start
        else:
          while next_morph()['end'] < ne['end']:
            pass
        end = next_morph()
        ne['end'] = end['id']

        target['named_entity'].append(ne)
      
      # add ner labels
      ner_labels.add(ne['label'])
  del modu
  ner_labels = sorted(list(ner_labels))

  # Sum up...
  victornlp = list(victornlp.values())
  for sent in victornlp: #cleanup begin and end in PoS
    for word in sent['pos']:
      for morph in word:
        morph.pop('begin')
        morph.pop('end')
  
  labels = {
    'pos_labels': pos_labels,
    'dp_labels': dp_labels,
    'ner_labels': ner_labels
  }

  print('data count: ', len(victornlp))
  
  print(json.dumps(victornlp[0], indent=4, ensure_ascii=False))
 
  train_len = int(0.8 * len(victornlp))
  dev_len = int(0.1 * len(victornlp))
  test_len = len(victornlp) - train_len - dev_len
  split = (train_len, dev_len, test_len)
  train, dev, test = tuple(random_split(victornlp, split))
  json.dump(list(train), train_file, indent=4, ensure_ascii = False)
  json.dump(list(dev), dev_file, indent=4, ensure_ascii = False)
  json.dump(list(test), test_file, indent=4, ensure_ascii = False)
  json.dump(labels, labels_file, indent=4, ensure_ascii = False)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Reformat Modu corpus(NIKL) into VictorNLP format')
  parser.add_argument('file-dir', type=str, help='Working directory that contains Modu corpus files')
  parser.add_argument('--dp', type=str, default='Modu_DP_raw.json', help='Name of DP data file')
  parser.add_argument('--pos', type=str, default='Modu_PoS_raw.json', help='Name of PoS data file')
  parser.add_argument('--ner', type=str, default='Modu_NER_raw.json', help='Name of NER data file')
  os.chdir(args.file_dir)
  with open(args.dp) as modu_dp_file, \
       open(args.pos) as modu_pos_file, \
       open(args.ner) as modu_ner_file, \
       open('VictorNLP_kor(Modu)_train.json', 'w') as train_file, \
       open('VictorNLP_kor(Modu)_dev.json', 'w') as test_file, \
       open('VictorNLP_kor(Modu)_test.json', 'w') as test_file, \
       open('VictorNLP_kor(Modu)_labels.json', 'w') as labels_file:
    modu_to_victornlp(modu_dp_file, modu_pos_file, modu_ner_file, train_file, dev_file, test_file, labels_file)
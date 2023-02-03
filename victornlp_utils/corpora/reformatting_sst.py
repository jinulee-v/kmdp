"""
@module reformatting_sst
Convert SST(Stanford Sentiment Tree) corpus(dependency tree version) to VictorNLP corpus format.
Is designed to reformat the file distributed in:
> https://github.com/ttpro1995/TreeLSTMSentiment
Execute ./fetch_and_preprocess.sh to obtain relevant files(download original SST and dependency-parse it).
"""

import os
import json
import argparse

from ..pos_tagger import pos_taggers

# If word is in the set, do not insert space.
word_concat_with_prev = set([
  '...', ':', ';', '%', '\'\'',     # exceptional cases
  '\'t', 'n\'t',
  '\'ll', '\'s', '\'d'
])
word_concat_with_next = set([
  '$', '#', '``'            # currency units
])

# Dictionary for raw text retrevial
punct = {
  '-LRB-': '(',
  '-RRB-': ')',
  '-LCB-': '{',
  '-RCB-': '}',
  '``'   : '\'',
  '\'\'' : '\'',
  '--'   : '-'
}

def sst_to_victornlp(sents_txt, sents_toks, rels_txt, dparents_txt, dlabels_txt, out_file, labels_file):
  raw_sents = sents_txt.readlines()
  tokens = sents_toks.readlines()
  rels = rels_txt.readlines()
  dparents = dparents_txt.readlines()
  dlabels = dlabels_txt.readlines()

  assert len(raw_sents) == len(tokens) == len(rels) == len(dparents) == len(dlabels)

  tokens = [line.split() for line in tokens]
  rels = [line.split() for line in rels]
  dparents = [line.split() for line in dparents]
  dlabels = [line.split() for line in dlabels]

  victornlp = []
  pos_labels = set()
  dp_labels = set()
  sentiment_labels = set()
  for raw_sent, token, rel, dparent, dlabel in zip(raw_sents, tokens, rels, dparents, dlabels):
    if not len(token) == len(rel) == len(dparent) == len(dlabel):
      print(len(token), len(rel), len(dparent), len(dlabel))
      continue
      return

    # Work with text_clean
    for symbol in word_concat_with_prev:
      raw_sent = raw_sent.replace(' '+symbol, symbol)
    for symbol in word_concat_with_next:
      raw_sent = raw_sent.replace(symbol+' ', symbol)
    for before, after in punct.items():
      raw_sent = raw_sent.replace(before, after)
    
    # Work with pos_tag

    pos = pos_taggers['english']([{
      'text': ' '.join(token),
      'word_count': len(token)
    }])[0]['pos']
    for morph in pos:
      pos_labels.add(morph['pos_tag'])

    # Work with dependency
    dependency = []
    for dep, head, label in zip(range(1, len(token)+1), dparent, rel):
      dependency.append({
        'dep': dep,
        'head': int(head),
        'label': label
      })
    for dp_label in rel:
      dp_labels.add(dp_label)
    
    # Work with pos tagger

    # Work with sentiment
    sentiment = []
    for head, sent_label in zip(range(1, len(token)+1), dlabel):
      if sent_label == '#':
        # Dependency span does not match original binarized constituents.
        continue
      sentiment.append({
        'head': head,
        'label': sent_label
      })
    for sentiment_label in dlabel:
      if sent_label == '#':
        # Dependency span does not match original binarized constituents.
        continue
      sentiment_labels.add(sentiment_label)
    
    victornlp.append({
      'text': ' '.join(token),
      'text_clean': raw_sent,
      'pos': pos,
      'word_count': len(token),
      'dependency': dependency,
      'sentiment': sentiment
    })
  
  labels = {
    'pos_labels': sorted(list(pos_labels)),
    'dp_labels': sorted(list(dp_labels)),
    'sentiment_labels': sorted(list(sentiment_labels))
  }

  json.dump(victornlp, out_file, ensure_ascii=False, indent=4)
  json.dump(labels, labels_file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Reformat SST(Stanford Sentiment Tree) Dependency corpus into VictorNLP format')
  parser.add_argument('file_dir', type=str, help='Working directory that contains SST files. refer to https://github.com/ttpro1995/TreeLSTMSentiment')
  args = parser.parse_args()

  os.chdir(args.file_dir)
  with open(os.path.join(args.file_dir, 'sents.txt')) as sents_txt, \
       open(os.path.join(args.file_dir, 'sents.toks')) as sents_toks, \
       open(os.path.join(args.file_dir, 'rels.txt')) as rels_txt, \
       open(os.path.join(args.file_dir, 'dparents.txt')) as dparents_txt, \
       open(os.path.join(args.file_dir, 'dlabels.txt')) as dlabels_txt, \
       open('VictorNLP_eng(SST).json', 'w') as out_file, \
       open('VictorNLP_eng(SST)_labels.json', 'w') as labels_file:
    sst_to_victornlp(sents_txt, sents_toks, rels_txt, dparents_txt, dlabels_txt, out_file, labels_file)
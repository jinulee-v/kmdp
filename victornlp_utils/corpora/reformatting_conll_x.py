"""
@module reformatting_conll_x
Convert CoNLL_X shared task corpus to VictorNLP corpus format.
Is designed to reformat the file distributed in:
> https://github.com/KhalilMrini/LAL-Parser
"""

import os, sys
import json

# If word is in the set, do not insert space.
pos_concat_with_prev = set([
  'POS',                    # possession ('s, ...s')
  '-RRB-', '-RCB-', '\'\'', # closing pairs
  ',', '.'                  # some puncts
])
# exceptional cases:
word_concat_with_prev = set([
  '...', ':', ';', '%',     # exceptional cases
  '\'t', 'n\'t',
  '\'ll', '\'s', '\'d'
])
pos_concat_with_next = set([
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

# colums for each information
DEP = 0
WORD = 1
POS = 3
HEAD = 6
LABEL = 7

def ptb_to_victornlp(ptb_dp_train, ptb_dp_dev, ptb_dp_test, train_file, dev_file, test_file, labels_file):
  
  # process train
  victornlp = []
  ptb = ptb_dp_train.readlines()
  pos_labels = set()
  dp_labels = set()
  for line in ptb:
    # skip empty lines
    line = line.strip()
    if not line:
      continue

    # basic preprocessing
    line = line.split('\t')
    line[DEP] = int(line[DEP])
    line[HEAD] = int(line[HEAD])
    if line[WORD] in punct:
      line[WORD] = punct[line[WORD]]

    # insert new sentence when meet initial word
    if line[DEP] == 1:
      victornlp.append({
        'text': '',
        'text_clean': '',
        'word_count': 0,
        'pos': [],
        'dependency': []
      })
      text = ''
      concat_next = True
    
    # if concat: do not insert space before current token
    concat = line[POS] in pos_concat_with_prev \
             or line[WORD] in word_concat_with_prev \
             or concat_next
    # if concat_next: do not insert space after current token
    concat_next = line[POS] in pos_concat_with_next
    
    victornlp[-1]['text'] += ('' if not victornlp[-1]['text'] else ' ') + line[WORD]
    victornlp[-1]['text_clean'] += ('' if concat else ' ') + line[WORD]
    victornlp[-1]['pos'].append({
      'id': line[DEP],
      'text': line[WORD],
      'pos_tag': line[POS]
    })
    victornlp[-1]['word_count'] = len(victornlp[-1]['pos'])
    victornlp[-1]['dependency'].append({
      'dep': line[DEP],
      'head': line[HEAD],
      'label': line[LABEL]
    })
    pos_labels.add(line[POS])
    dp_labels.add(line[LABEL])

  print('train data count: ', len(victornlp))
  del ptb
  pos_labels = sorted(list(pos_labels))
  dp_labels = sorted(list(dp_labels))
  
  labels = {
    'pos_labels': pos_labels,
    'dp_labels': dp_labels
  }
  print(json.dumps(victornlp[0], indent=4, ensure_ascii=False))
  json.dump(victornlp, train_file, indent=4, ensure_ascii = False)
  json.dump(labels, labels_file, indent=4, ensure_ascii = False)
 
  # process dev
  victornlp = []
  ptb = ptb_dp_dev.readlines()
  for line in ptb:
    # skip empty lines
    line = line.strip()
    if not line:
      continue

    # basic preprocessing
    line = line.split('\t')
    line[DEP] = int(line[DEP])
    line[HEAD] = int(line[HEAD])
    if line[WORD] in punct:
      line[WORD] = punct[line[WORD]]

    # insert new sentence when meet initial word
    if line[DEP] == 1:
      victornlp.append({
        'text': '',
        'text_clean': '',
        'word_count': 0,
        'pos': [],
        'dependency': []
      })
      text = ''
      concat_next = True
    # if concat: do not insert space before current token
    concat = line[POS] in pos_concat_with_prev \
             or line[WORD] in word_concat_with_prev \
             or concat_next
    # if concat_next: do not insert space after current token
    concat_next = line[POS] in pos_concat_with_next
    
    victornlp[-1]['text'] += ('' if not victornlp[-1]['text'] else ' ') + line[WORD]
    victornlp[-1]['text_clean'] += ('' if concat else ' ') + line[WORD]
    victornlp[-1]['pos'].append({
      'id': line[DEP],
      'text': line[WORD],
      'pos_tag': line[POS]
    })
    victornlp[-1]['word_count'] = len(victornlp[-1]['pos'])
    victornlp[-1]['dependency'].append({
      'dep': line[DEP],
      'head': line[HEAD],
      'label': line[LABEL]
    })

  print('dev data count: ', len(victornlp))
  del ptb
  json.dump(victornlp, dev_file, indent=4, ensure_ascii = False)

  # process test
  victornlp = []
  ptb = ptb_dp_test.readlines()
  for line in ptb:
    # skip empty lines
    line = line.strip()
    if not line:
      continue

    # basic preprocessing
    line = line.split('\t')
    line[DEP] = int(line[DEP])
    line[HEAD] = int(line[HEAD])
    if line[WORD] in punct:
      line[WORD] = punct[line[WORD]]

    # insert new sentence when meet initial word
    if line[DEP] == 1:
      victornlp.append({
        'text': '',
        'text_clean': '',
        'word_count': 0,
        'pos': [],
        'dependency': []
      })
      text = ''
      concat_next = True
    
    # if concat: do not insert space before current token
    concat = line[POS] in pos_concat_with_prev \
             or line[WORD] in word_concat_with_prev \
             or concat_next
    # if concat_next: do not insert space after current token
    concat_next = line[POS] in pos_concat_with_next
    
    victornlp[-1]['text'] += ('' if not victornlp[-1]['text'] else ' ') + line[WORD]
    victornlp[-1]['text_clean'] += ('' if concat else ' ') + line[WORD]
    victornlp[-1]['pos'].append({
      'id': line[DEP],
      'text': line[WORD],
      'pos_tag': line[POS]
    })
    victornlp[-1]['word_count'] = len(victornlp[-1]['pos'])
    victornlp[-1]['dependency'].append({
      'dep': line[DEP],
      'head': line[HEAD],
      'label': line[LABEL]
    })

  print('test data count: ', len(victornlp))
  del ptb
  json.dump(victornlp, test_file, indent=4, ensure_ascii = False)
  


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Reformat PTB(CoNLL-X style) into VictorNLP format')
  parser.add_argument('file-dir', type=str, help='Working directory that contains PTB corpus files')
  parser.add_argument('--train', type=str, default='PTB_DP_raw_train.txt', help='Name of train data file')
  parser.add_argument('--dev', type=str, default='PTB_DP_raw_dev.txt', help='Name of dev data file')
  parser.add_argument('--test', type=str, default='PTB_DP_raw_test.txt', help='Name of test data file')
  os.chdir(args.file_dir)
  with open('PTB_DP_raw_train.txt') as ptb_dp_train, \
       open('PTB_DP_raw_dev.txt') as ptb_dp_dev, \
       open('PTB_DP_raw_test.txt') as ptb_dp_test, \
       open('VictorNLP_eng(PTB)_train.json', 'w') as train_file, \
       open('VictorNLP_eng(PTB)_dev.json', 'w') as dev_file, \
       open('VictorNLP_eng(PTB)_test.json', 'w') as test_file, \
       open('VictorNLP_eng(PTB)_labels.json', 'w') as labels_file:
    ptb_to_victornlp(ptb_dp_train, ptb_dp_dev, ptb_dp_test, train_file, dev_file, test_file, labels_file)
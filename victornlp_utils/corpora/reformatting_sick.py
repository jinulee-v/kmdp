"""
@module reformatting_sick
Convert SICK(Stanford Sentiment Tree) corpus(dependency tree version) to VictorNLP corpus format.
Is designed to reformat the file distributed in:
> https://github.com/dasguptar/treelstm.pytorch
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

def sick_to_victornlp(sents_txt, sents_toks, rels_txt, dparents_txt, sim_txt, ent_txt, out_file, pairinfo_file, labels_file):
  raw_sents = sents_txt.readlines()
  tokens = sents_toks.readlines()
  rels = rels_txt.readlines()
  dparents = dparents_txt.readlines()
  sims = sim_txt.readlines()
  ents = ent_txt.readlines()

  assert len(raw_sents) == len(tokens) == len(rels) == len(sims) == len(ents) == len(dparents)

  tokens = [line.split() for line in tokens]
  rels = [line.split() for line in rels]
  dparents = [line.split() for line in dparents]
  ents = [line.strip() for line in ents]

  victornlp = []
  pos_labels = set()
  dp_labels = set()
  nli_labels = set(ents)
  for raw_sent, token, rel, dparent in zip(raw_sents, tokens, rels, dparents):
    if not len(token) == len(rel) == len(dparent):
      print(len(token), len(rel), len(dparent))
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
    sts_nli = []
    for sts, nli in zip(sims, ents):
      sts_nli.append({
        'sts': round(float(sts), 2),
        'nli': nli
      })
    
    victornlp.append({
      'text': ' '.join(token),
      'text_clean': raw_sent.strip(),
      'word_count': len(token),
      'pos': pos,
      'dependency': dependency
    })
  
  labels = {
    'pos_labels': sorted(list(pos_labels)),
    'dp_labels': sorted(list(dp_labels)),
    'nli_labels': sorted(list(nli_labels))
  }

  json.dump(victornlp, out_file, ensure_ascii=False, indent=4)
  json.dump(sts_nli, pairinfo_file, ensure_ascii=False, indent=4)
  json.dump(labels, labels_file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Reformat SST(Stanford Sentiment Tree) Dependency corpus into VictorNLP format')
  parser.add_argument('file_dir', type=str, help='Working directory that contains SST files. refer to https://github.com/ttpro1995/TreeLSTMSentiment')
  args = parser.parse_args()

  os.chdir(args.file_dir)
  with open(os.path.join(args.file_dir, 'a.txt')) as sents_txt, \
       open(os.path.join(args.file_dir, 'a.toks')) as sents_toks, \
       open(os.path.join(args.file_dir, 'a.rels')) as rels_txt, \
       open(os.path.join(args.file_dir, 'a.parents')) as dparents_txt, \
       open(os.path.join(args.file_dir, 'sim.txt')) as sim_txt, \
       open(os.path.join(args.file_dir, 'ent.txt')) as ent_txt, \
       open('VictorNLP_eng(SICK).a.json', 'w') as out_file, \
       open('VictorNLP_eng(SICK).pairinfo.json', 'w') as pairinfo_file, \
       open('VictorNLP_eng(SICK)_labels.json', 'w') as labels_file:
    sick_to_victornlp(sents_txt, sents_toks, rels_txt, dparents_txt, sim_txt, ent_txt, out_file, pairinfo_file, labels_file)
  with open(os.path.join(args.file_dir, 'b.txt')) as sents_txt, \
       open(os.path.join(args.file_dir, 'b.toks')) as sents_toks, \
       open(os.path.join(args.file_dir, 'b.rels')) as rels_txt, \
       open(os.path.join(args.file_dir, 'b.parents')) as dparents_txt, \
       open(os.path.join(args.file_dir, 'sim.txt')) as sim_txt, \
       open(os.path.join(args.file_dir, 'ent.txt')) as ent_txt, \
       open('VictorNLP_eng(SICK).b.json', 'w') as out_file, \
       open('VictorNLP_eng(SICK).pairinfo.json', 'w') as pairinfo_file, \
       open('VictorNLP_eng(SICK)_labels.json', 'w') as labels_file:
    sick_to_victornlp(sents_txt, sents_toks, rels_txt, dparents_txt, sim_txt, ent_txt, out_file, pairinfo_file, labels_file)
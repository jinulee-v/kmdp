"""
@module reformatting_korsts
Convert Korean STS(Sentence Text Similarity) corpus to VictorNLP corpus format.
Is designed to reformat the file distributed in:
> https://github.com/kakaobrain/KorNLUDatasets
"""

import argparse
from ..pos_tagger import pos_taggers
import json

def main(args):
  with open(args.src_file, 'r', encoding='UTF-8') as file:
    lines = file.readlines()
  
  api = KhaiiiApi()

  a = []
  b = []
  scores = []
  for line in lines[1:]:
    line = line.split('\t')

    # a.json
    text = line[5].strip()
    pos = pos_taggers['korean']([{'text': text, 'word_count': len(text.split(' '))}])['pos']
    if pos[-1][-1]['id'] > 510:
      continue
    a.append({
      'text': text,
      'word_count': len(text.split(' ')),
      'pos': pos
    })

    # b.json
    text = line[6].strip()
    pos = pos_taggers['korean']([{'text': text, 'word_count': len(text.split(' '))}])['pos']
    if pos[-1][-1]['id'] > 510:
      continue
    b.append({
      'text': text,
      'word_count': len(text.split(' ')),
      'pos': pos
    })

    # scores.json
    scores.append({"sts": float(line[4])})
  
  
  with open(args.dst_header + '.a.json', 'w', encoding='UTF-8') as file:
    json.dump(a, file, indent=4, ensure_ascii=False)
  with open(args.dst_header + '.b.json', 'w', encoding='UTF-8') as file:
    json.dump(b, file, indent=4, ensure_ascii=False)
  with open(args.dst_header + '.pairinfo.json', 'w', encoding='UTF-8') as file:
    json.dump(scores, file, indent=4)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--src-file')
  parser.add_argument('--dst-header')
  args = parser.parse_args()

  main(args)
"""
Generates ***_labels.json for KMDP.
"""

import argparse
import json

def main(args):
  with open(args.src_file, 'r', encoding="UTF-8") as file:
    inputs = json.load(file)
  
  pos_labels = set()
  dp_labels = set()
  for sentence in inputs:
    if not 'kmdp' in sentence:
      continue
    for wp in sentence['pos']:
      for morph in wp:
        pos_labels.add(morph['pos_tag'])
    for arc in sentence['kmdp']:
      dp_labels.add(arc['label'])
  result = {
    'pos_labels' : sorted(list(pos_labels)),
    'dp_labels' : sorted(list(dp_labels)),
  }
  with open(args.dst_file, 'w', encoding="UTF-8") as file:
    json.dump(result, file, indent=4)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Generates labels.json for KMDP")
  parser.add_argument('--src_file', type=str, help="JSON file, output of generate_kmdp.py")
  parser.add_argument('--dst_file', type=str, help="JSON output file, VictorNLP label format")

  args = parser.parse_args()
  main(args)
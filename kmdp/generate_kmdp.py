"""
@module generate_kmdp
Generates KMDP from DP & PoS tagging results.
"""

import argparse
import json
from tqdm import tqdm

from dependency import kmdp_rules, kmdp_generate
from dependency.lexicals import label2head
from dependency.interface import KMDPGenerateException

def generate_rule_based_KMDP(sentence):
  """
  @brief generate KMDP of the sentence.

  sentence = {
    'text': (raw text),
    'word_count': (N of word-phrases) /*== len(text.split())*/ ,
    'pos': [
      [
        {
          'id': (Nst morpheme in sentence, starts from 1)
          'text': (raw form of the morpheme),
          'pos_tag': (PoS tag)
        }, ... 
      ], ...
    ] /* len(pos) == word_count */ ,
    'dependency': [
      {
        'dep': (dependent is Nth word-phrase)
        'head': (head is Nth word-phrase)
        'label': (dependency label)
      }
    ] /*len(dependency) = word_count*/
  }
  """
  dp = sentence['dependency']
  pos = sentence['pos']
  kmdp = []

  for i, (arc, dep_wp) in enumerate(zip(dp, pos), 1):
    assert i == arc['dep']
    if arc['head'] == 0:
      # ROOT
      head_wp = [
        {
          'id': 0,
          'text': '[ROOT]',
          'pos_tag': '[ROOT]'
        }
      ]
    else:
      head_wp = pos[arc['head'] - 1] # dp indexing starts from 1
    
    for j in range(len(dep_wp)):
      # for each morpheme,

      for rule in kmdp_rules:
        kmdp_arc = kmdp_generate(rule, dep_wp, j, head_wp, arc['label'])
        if kmdp_arc:
          # if any valid result,
          if dep_wp[j]['pos_tag'] not in label2head['all_heads']:
            raise KMDPGenerateException(rule, 'Non-head morphemes should not have heads.', dep_wp, j, head_wp, arc['label'])
          break
      
      if not kmdp_arc:
        # Only non-head morpheme reach here
        # (default_* rules filter all heads)
        if dep_wp[j]['pos_tag'] in label2head['all_heads']:
          raise KMDPGenerateException('-', 'Head morphemes must have heads.\nPossible cause: default_* rules did not correctly sieve all head morphemes.', dep_wp, j, head_wp, arc['label'])
        continue
        
      # If any valid result, add to list.
      kmdp.append(kmdp_arc)
  
  sentence['kmdp'] = kmdp
  return sentence

###############################################################################

def format_morph(morph):
  """
  Format morphemes to compact string.
  """
  if morph['text'] == '[ROOT]':
    return '[ROOT]'
  return morph['text'] + '/' + morph['pos_tag']
      
def main(args):
  # Remove excluded rules
  if args.exclude:
    for alias in args.exclude:
      kmdp_rules.remove(alias)

  with open(args.src_file, 'r', encoding='UTF-8') as src_file:
    inputs = json.load(src_file)
  
  errors = []

  for sentence in tqdm(inputs):
    try:
      generate_rule_based_KMDP(sentence)
    except KMDPGenerateException as e:
      errors.append({
        'sentence': sentence,
        'environment': e.environment
      })
      if args.interactive:
        interactive_ui(e)
  
  for error in errors:
    print(error['sentence']['text'])
    print('  rule:     {}'.format(error['environment']['rule']))
    print('  msg:      {}'.format(error['environment']['msg']))
    print('  dep_wp:   {}'.format(' + '.join([format_morph(morph) for morph in error['environment']['dep_wp']])))
    print('  dep_wp_i: {}'.format(error['environment']['dep_wp_i']))
    print('  head_wp:  {}'.format(' + '.join([format_morph(morph) for morph in error['environment']['head_wp']])))
    print('  dp_label: {}'.format(error['environment']['dp_label']))
    print('')

  if args.simple:
    # write into simple CoNLL-X style format(for visualization)
    kmdp_string = ''
    for sentence in inputs:
      if 'kmdp' not in sentence:
        continue
      text = sentence['text']
      pos = ' '.join(['+'.join([format_morph(morph) for morph in word]) for word in sentence['pos']])
      kmdp_arcs = ''
      pos_flatten = [{'id': 0, 'text': '[ROOT]', 'pos_tag': '[ROOT]'}]
      for word in sentence['pos']:
        pos_flatten.extend(word)
      for kmdp_arc in sentence['kmdp']:
        kmdp_arcs += '{}\t->\t{}\t{: <5}\n'.format(format_morph(pos_flatten[kmdp_arc['dep']]), format_morph(pos_flatten[kmdp_arc['head']]), kmdp_arc['label'])
      
      # Update final string
      kmdp_string += '; ' + text + '\n# ' + pos + '\n' + kmdp_arcs + '\n'


    if args.dst_file == '-':
      print(kmdp_string)
    else:
      with open(args.dst_file, 'w', encoding='UTF-8') as dst_file:
        dst_file.write(kmdp_string)
  else:
    if args.dst_file == '-':
      print(json.dumps(inputs, indent=4, ensure_ascii=False))
    else:
      with open(args.dst_file, 'w', encoding='UTF-8') as dst_file:
        json.dump(inputs, dst_file, indent=4, ensure_ascii=False)


def cli_main():
  parser = argparse.ArgumentParser(description='Generate KMDP out of Korean corpus, given DP and PoS-tagging results.')

  # File arguments
  parser.add_argument('--src-file', type=str, help='File to load and parse')
  parser.add_argument('--dst-file', type=str, default='-', help='File to store results. Default: stdout')

  # Options to detect and fix unexpected situations.
  parser.add_argument('--interactive', action='store_true', help='if any error is raised, interactively parse.')
  parser.add_argument('--error-file', type=str, help='if any error is raised, log situations.')
  parser.add_argument('--exclude', '-x', type=str, nargs='*', action='append', help='Rule name(alias) to exclude from current run.')
  parser.add_argument('--simple', action='store_true', help='Only print KMDP results, not full VictorNLP format corpus.')

  args = parser.parse_args()
  main(args)


if __name__ == '__main__':
  cli_main()

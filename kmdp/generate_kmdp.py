"""
@module generate_kmdp
Generates KMDP from DP & PoS tagging results.
"""

import argparse
import json
import logging
from collections import Counter
from datetime import datetime
from tqdm import tqdm

from dependency import kmdp_rules, kmdp_generate
from dependency.lexicals import label2head
from dependency.interface import KMDPGenerateException, get_doc, find_dominated_rules, is_local

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

      # kmdp_arc: arc information determined
      # applied_rule: rule applied
      # safe_rules: dominees of the applied rule
      kmdp_arc = None
      applied_rule = None
      safe_rules = []
      for rule in kmdp_rules:
        try:
          if is_local(rule):
            kmdp_arc_temp = kmdp_generate(rule, dep_wp, j, head_wp, arc['label'])
          else:
            kmdp_arc_temp = kmdp_generate(rule, sentence, i, j)
        except KMDPGenerateException as e:
          if not applied_rule:
            raise e

        if not kmdp_arc:
          kmdp_arc = kmdp_arc_temp
        if kmdp_arc_temp and not applied_rule:
          # if any valid result,
          if dep_wp[j]['pos_tag'] not in label2head['all_heads']:
            raise KMDPGenerateException(rule, 'Non-head morphemes should not have heads.', dep_wp, j, head_wp, arc['label'])
          # update value
          applied_rule = rule
          safe_rules = find_dominated_rules(rule)
        elif kmdp_arc_temp and applied_rule and rule not in safe_rules:
          # If two un-dominated rules collide:
          raise KMDPGenerateException(rule, 'Rule collision with {}'.format(applied_rule), dep_wp, j, head_wp, arc['label'])
      
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
  # Configurate loggers
  error_handler = logging.FileHandler(args.error_file, 'w', encoding='UTF-8')
  error_logger = logging.getLogger('error')
  error_logger.addHandler(error_handler)
  error_logger.setLevel(logging.INFO)

  if args.dst_file == '-':
    result_handler = logging.StreamHandler()
  else:
    result_handler = logging.FileHandler(args.dst_file, 'w', encoding='UTF-8')
  result_logger = logging.getLogger('result')
  result_logger.addHandler(result_handler)
  result_logger.setLevel(logging.INFO)

  # Remove excluded rules
  if args.exclude:
    for alias in args.exclude:
      kmdp_rules.remove(alias[0])
  
  print('-'*80)
  print('Applied rules\n')
  for i, rule in enumerate(kmdp_rules):
    print(str(i+1) + '. ' + rule + ':')
    print(get_doc(rule))
  print('-'*80)

  with open(args.src_file, 'r', encoding='UTF-8') as src_file:
    inputs = json.load(src_file)
  
  errors = []
  error_rules = Counter()

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
    # except Exception as e:
    #   print(e)
    #   return None
  
  for error in errors:
    error_logger.info(error['sentence']['text'])
    error_logger.info('  rule:     %s', error['environment']['rule'])
    error_logger.info('  msg:      %s', error['environment']['msg'])
    error_logger.info('  dep_wp:   %s', ' + '.join([format_morph(morph) for morph in error['environment']['dep_wp']]))
    error_logger.info('  dep_wp_i: %d', error['environment']['dep_wp_i'])
    error_logger.info('  head_wp:  %s', ' + '.join([format_morph(morph) for morph in error['environment']['head_wp']]))
    error_logger.info('  dp_label: %s', error['environment']['dp_label'])
    error_logger.info('')
    error_rules.update([error['environment']['rule']])
  
  error_logger.info('Stats')
  error_logger.info('  total errors: %d / %d', len(errors), len(inputs))

  error_logger.info('')
  error_logger.info('Errors per rules:')
  for rule, count in error_rules.most_common():
    error_logger.info('  %s : %d / %d', rule, count, len(errors))

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

    result_logger.info(kmdp_string)

  elif args.brat:
    # write into brat format(for brat visualization)
    divide_wp = '  ▁  '
    divide_morph = '  '

    # txt_i: Current character index
    txt_i = 0
    kmdp_string = ''
    # span_i, arc_i: Current annotation index
    span_i = 1
    arc_i = 1
    annotation = []

    page_i = 1
    for sent_i, sentence in enumerate(inputs):
      # Skip if generated error
      if 'kmdp' not in sentence:
        continue
    
      # morph_info: mapping between morph['id'] and span_i

      pos = [[{'id': 0, 'text': '[ROOT]', 'pos_tag': '[ROOT]'}]] + sentence['pos']

      morph_info = {}
      for word in pos:
        for morph in word:
          # Update raw text
          kmdp_string += morph['text']

          # Update span info
          annotation.append('T{}\t{} {} {}\t{}'.format(span_i, morph['pos_tag'], txt_i, txt_i+len(morph['text']), morph['text']))
          morph_info[morph['id']] = 'T' + str(span_i)

          # Update indices
          txt_i += len(morph['text'])
          span_i += 1

          # Divider between morphemes
          if morph != word[-1]:
            kmdp_string += divide_morph
            txt_i += len(divide_morph)

        # Divider between word-phrases
        if word != sentence['pos'][-1]:
          kmdp_string += divide_wp
          txt_i += len(divide_wp)
      
      # Update arc info
      for arc in sentence['kmdp']:
        annotation.append('R{}\t{} Arg1:{} Arg2:{}'.format(arc_i, arc['label'], morph_info[arc['head']], morph_info[arc['dep']]))
        arc_i += 1

      # Line-breakers
      kmdp_string += '\n'
      txt_i += 1
    
      if sent_i % 20 == 19 or sent_i == len(inputs)-1:
        # Pad page index with zeros
        page_str_i = '0' * (len(str((len(inputs)-1)//20+1)) - len(str(page_i))) + str(page_i)
        with open(args.brat+'/result_{}.txt'.format(page_str_i), 'w', encoding='UTF-8') as file:
          file.write(kmdp_string)

        with open(args.brat+'/result_{}.ann'.format(page_str_i), 'w', encoding='UTF-8') as file:
          file.write('\n'.join(annotation))
        txt_i = 0
        kmdp_string = ''
        span_i = 1
        arc_i = 1
        annotation = []
        page_i += 1
      
  else:
    result_logger.info(json.dumps(inputs, indent=4, ensure_ascii=False))


def cli_main():
  parser = argparse.ArgumentParser(description='Generate KMDP out of Korean corpus, given DP and PoS-tagging results.')

  # File arguments
  parser.add_argument('--src-file', type=str, help='File to load and parse')
  parser.add_argument('--dst-file', type=str, default='-', help='File to store results. Default: stdout')

  # Options to detect and fix unexpected situations.
  parser.add_argument('--interactive', action='store_true', help='NOT IMPLEMENTED If any error is raised, interactively parse.')
  parser.add_argument('--error-file', type=str, default='error_'+datetime.now().strftime('%y%m%d_%H%M%S')+'.log', help='if any error is raised, log errors to file.')
  parser.add_argument('--exclude', '-x', type=str, nargs='*', action='append', help='Rule name(alias) to exclude from current run.')
  parser.add_argument('--simple', action='store_true', help='Only print KMDP results, not full VictorNLP format corpus.')
  parser.add_argument('--brat', type=str, help='Directory to brat/data. Print KMDP results as `brat` format.')

  args = parser.parse_args()

  if args.brat:
    assert not args.simple and 'Cannot select two output formats at once!'
    assert args.brat.endswith('/data') and 'Enter proper brat/data directory!'

  main(args)


if __name__ == '__main__':
  cli_main()

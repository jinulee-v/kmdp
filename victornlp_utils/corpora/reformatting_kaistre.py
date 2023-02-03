"""
@module reformatting_kaistre
Convert KAIST Relation Extraction corpus to VictorNLP corpus format.
Is designed to reformat the file distributed in:
> https://github.com/machinereading/bert-ko-re/tree/master/ko_re_data
"""

import argparse
import json
from time import sleep
from tqdm import tqdm

from ..pos_tagger import pos_taggers
from ..utils.dbpedia_query import entity_tag_ontology

def kaistre_to_victornlp(args):
  with open(args.src_file, 'r', encoding='UTF-8') as file:
    lines = file.readlines()

  a = []
  ontology_tags = {}
  ontology_tagset = set()
  relation_tagset = set()
  for line in tqdm(lines):
    line = line.split('\t')
    text = line[3].strip()
    # remove hidden unicode blank spaces
    text = text.replace(u'\u00A0', ' ').replace(u'\u2009', ' ')

    e1 = line[1]; e2 = line[2]
    relation = line[0]

    # find entity
    e1_index = text.find('<e1>')
    e2_index = text.find('<e2>')
    assert e1_index != -1 and e2_index != -1 # tag must exist

    # ontology tagging
    if e1 not in ontology_tags:
      success = False
      count = 5
      while not success and count > 0:
        try:
          ontology_tags[e1] = entity_tag_ontology(e1, 'ko')
          success = True
        except Exception as e:
          print(e)
          sleep(1)
          count -= 1
      if count == 0:
        ontology_tags[e1] = []

      for tag in ontology_tags[e1]:
        ontology_tagset.add(tag)
    if e2 not in ontology_tags:
      success = False
      count = 5
      while not success and count > 0:
        try:
          ontology_tags[e2] = entity_tag_ontology(e2, 'ko')
          success = True
        except Exception as e:
          print(e)
          count -= 1
          sleep(1)
      if count == 0:
        ontology_tags[e2] = []
      for tag in ontology_tags[e2]:
        ontology_tagset.add(tag)

    if e1_index < e2_index:
      e2_index -= 9
    else:
      e1_index -= 9
    e1_info = \
        {
          'text': e1,
          'label': ontology_tags[e1],
          'begin': e1_index,
          'end': e1_index+len(e1)
        }
    e2_info = \
        {
          'text': e2,
          'label': ontology_tags[e2],
          'begin': e2_index,
          'end': e2_index+len(e2)
        }
    
    # remove entity boundary tags
    text = text.replace('<e1>', '').replace('</e1>', '').replace('<e2>', '').replace('</e2>', '')

    if len(a) == 0 or text != a[-1]['text']:
      # New sentence information begins
      named_entity = []
      relations = []
      try:
        pos = pos_taggers['korean']([{'text': text, 'word_count': len(text.split(' '))}])[0]['pos']
      except Exception as e:
        print('\nError occured in Khaiii:')
        print(e)
        continue
      
      # Manual correction for PoS tagging
      for i, wp in enumerate(pos):
        if len(wp) == 2 and wp[0]['text'] == '가' and wp[1]['text'] == '아':
          pos[i] = [{
            'id': None,
            'text': '가',
            'pos_tag': 'JKS'
          }]
      # Renumbering IDs after modification
      id = 1
      for wp in pos:
        for morph in wp:
          morph['id'] = id
          id += 1
      if id > 200:
        continue

      a.append({
        'text': text,
        'word_count': len(text.split(' ')),
        'pos': pos,
        'named_entity': named_entity,
        'relation': relations
      })
    else:
      named_entity = a[-1]['named_entity']
      relations = a[-1]['relation']
    
    e1_id = None
    e2_id = None
    i = 0
    for i in range(len(named_entity)):
      if named_entity[i]['begin'] == e1_info['begin']:
        e1_id = i
        break
    if e1_id is None:
      named_entity.append(e1_info)
      e1_id = len(named_entity) - 1
    i = 0
    for i in range(len(named_entity)):
      if named_entity[i]['begin'] == e2_info['begin']:
        e2_id = i
        break
    if e2_id is None:
      named_entity.append(e2_info)
      e2_id = len(named_entity) - 1
        
    # Add relation
    relation_tagset.add(relation)
    # <<e2>> is <<label>> of <<e1>>
    relations.append(
      {
        'subject': e2_id,
        'predicate': e1_id,
        'label': relation
      }
    )
        
  
  with open(args.dst_file, 'w', encoding='UTF-8') as file:
    json.dump(a, file, indent=4, ensure_ascii=False)
  with open(args.dst_file.replace('.json', '.labels.json'), 'w', encoding='UTF-8') as file:
    json.dump({
      'entity_type_labels': sorted(list(ontology_tagset)),
      'relation_labels': sorted(list(relation_tagset))
    }, file, indent=4, ensure_ascii=False)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--src-file')
  parser.add_argument('--dst-file')
  args = parser.parse_args()

  kaistre_to_victornlp(args)
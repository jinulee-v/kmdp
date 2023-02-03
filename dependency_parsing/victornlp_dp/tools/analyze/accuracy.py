"""
@module analyze
Tools for analyzong the parser behavior such as accuracy().
"""

from . import register_analysis_fn

from ...kmdp.lexicals import _all_heads

@register_analysis_fn('accuracy')
def analyze_accuracy(inputs):
  """
  Calculates accuracy.
  
  @param inputs List of dictionaries. Refer to 'dataset.py'' for more details.
  
  @return Dictionary with keys 'uas' and 'las' in percentage(rounded for 4 digits).
  """
  total = 0
  unlabel = 0
  label = 0
  for input in inputs:
    assert 'dependency' in input
    assert 'dependency_predict' in input

    for golden, predict in zip(input['dependency'], input['dependency_predict']):
      if predict['dep'] != golden['dep']:
        continue
      total += 1
      if predict['head'] == golden['head']:
        unlabel += 1
        if predict['label'] == golden['label']:
          label += 1
  return {'uas': round(unlabel/total*100, 2), 'las': round(label/total*100, 2)}


@register_analysis_fn('accuracy_per_label')
def analyze_accuracy_per_label(inputs):
  """
  Calculates accuracy.
  
  @param inputs List of dictionaries. Refer to 'dataset.py' for more details.
  
  @return Dictionary containing label and accuracy(UAS) pair.
  """
  total = {}
  unlabel = {}
  for input in inputs:
    assert 'dependency' in input
    assert 'dependency_predict' in input

    for golden, predict in zip(input['dependency'], input['dependency_predict']):
      if predict['dep'] != golden['dep']:
        continue
      
      if golden['label'] not in total:
        total[golden['label']] = 0
        unlabel[golden['label']] = 0
      total[golden['label']] += 1
      if predict['head'] == golden['head']:
        unlabel[golden['label']] += 1

  return {label:unlabel[label]/total[label] for label in total.keys()}


@register_analysis_fn('accuracy_per_distance')
def analyze_accuracy_per_distance(inputs):
  """
  Calculates accuracy.
  
  @param inputs List of dictionaries. Refer to 'dataset.py' for more details.
  
  @return Dictionary containing label and accuracy(UAS) pair.
  """
  total = {}
  unlabel = {}
  for input in inputs:
    assert 'dependency' in input
    assert 'dependency_predict' in input

    for golden, predict in zip(input['dependency'], input['dependency_predict']):
      if predict['dep'] != golden['dep']:
        continue
      
      if golden['head'] == 0:
        key = 'ROOT'
      else:
        key = abs(golden['head'] - golden['dep'])
        for wp in input['pos']:
          for morph in wp:
            if min(golden['head'], golden['dep']) < morph['id'] < max(golden['head'], golden['dep']):
              if morph['pos_tag'] not in _all_heads:
                key -= 1

      if key not in total:
        total[key] = 0
        unlabel[key] = 0
      total[key] += 1
      if predict['head'] == golden['head']:
        unlabel[key] += 1

  return {key:unlabel[key]/total[key] for key in total.keys()}

@register_analysis_fn('accuracy_semi_las')
def analyze_accuracy_same_head_pos(inputs):
  """
  Calculates semi-LAS accuracy.
  semi-LAS: accuracy for 1. correct label 2. correct head part-of-speech(not the correct head but same label)
  
  @param inputs List of dictionaries. Refer to 'dataset.py' for more details.
  
  @return Dictionary containing semi-LAS for each label.
  """
  total = {}
  semi_labeled = {}
  for input in inputs:
    assert 'dependency' in input
    assert 'dependency_predict' in input

    for golden, predict in zip(input['dependency'], input['dependency_predict']):
      if predict['dep'] != golden['dep']:
        continue
      
      key = golden['label']

      if predict['label'] == golden['label']:
        # find head label
        golden_pos = predict_pos = None
        for arc in input['dependency']:
          if arc['dep'] == golden['head']:
            golden_pos = arc['label'].split('_')[0]
          if arc['dep'] == predict['head']:
            predict_pos = arc['label'].split('_')[0]

        if key not in total:
          total[key] = 0
          semi_labeled[key] = 0
        total[key] += 1
        if golden_pos is not None and predict_pos is not None:
          # not ROOT dependency
          if golden_pos == predict_pos:
            semi_labeled[key] += 1
        else:
          # ROOT dependency
          if predict['head'] == 0:
            semi_labeled[key] += 1

  return {key:semi_labeled[key]/total[key] for key in total.keys()}
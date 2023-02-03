"""
@module analyze
Tools for analyzong the parser behavior such as accuracy().
"""

from . import register_analysis_fn

@register_analysis_fn('mtre')
def analyze_mtre_basic(inputs):
  """
  Calculates UAS for DP; recovery for entity type labeling, and Accuracy for relation labeling.
  
  @param inputs List of dictionaries. Refer to 'dataset.py'' for more details.
  
  @return Dictionary with accuracy(total) and per-label in percentage(rounded for 4 digits).
  """
  dp_correct = 0
  dp_total = 0
  
  etl_correct = 0
  etl_total = 0
  etl_labels = {}
  
  re_correct = 0
  re_total = 0
  re_labels = {}

  for input in inputs:
    # UAS
    assert 'dependency' in input and 'dependency_predict' in input
    for arc_golden in input['dependency']:
      for arc_predict in input['dependency_predict']:
        if arc_golden['dep'] == arc_predict['dep']:
          dp_total += 1
          if arc_golden['head'] == arc_predict['head']:
            dp_correct += 1
    
    # Entity Type labeling
    for entity in input['named_entity']:
      assert 'label' in entity and 'label_predict' in entity
      etl_total += len(entity['label'])
      for predict in entity['label_predict']:
        if predict in entity['label']:
          etl_correct += 1
        if predict not in etl_labels:
          etl_labels[predict] = 0
        etl_labels[predict] += 1
    
    # Relation Extraction
    for relation in input['relation']:
      if 'label' in relation and 'label_predict' in relation:
        re_total += 1

        if relation['label_predict'] not in re_labels:
          re_labels[relation['label_predict']] = 0
        re_labels[relation['label_predict']] += 1

        if relation['label'] == relation['label_predict']:
          re_correct += 1

  dp_uas = dp_correct / dp_total * 100
  etl_recovery = etl_correct / etl_total * 100

  if re_total > 0:
    re_acc = re_correct / re_total * 100

    return {
      'dependency': {
        'UAS': round(dp_uas, 2)
      },
      'entity_type_label': {
        'recovery': round(etl_recovery, 2)
      },
      'relation_extraction': {
        'accuracy': round(re_acc, 2)
      }
    }
  else:
    return {
      'dependency': {
        'UAS': round(dp_uas, 2)
      },
      'entity_type_label': {
        'recovery': round(etl_recovery, 2)
      }
    }


@register_analysis_fn('accuracy')
def analyze_accuracy(inputs):
  """
  Calculates UAS for DP; recovery for entity type labeling, and Accuracy for relation labeling.
  
  @param inputs List of dictionaries. Refer to 'dataset.py'' for more details.
  
  @return Dictionary with accuracy(total) and per-label in percentage(rounded for 4 digits).
  """
  re_correct = 0
  re_total = 0
  re_labels = {}

  for input in inputs:
    
    # Relation Extraction
    for relation in input['relation']:
      if 'label' in relation and 'label_predict' in relation:
        re_total += 1

        if relation['label_predict'] not in re_labels:
          re_labels[relation['label_predict']] = 0
        re_labels[relation['label_predict']] += 1

        if relation['label'] == relation['label_predict']:
          re_correct += 1

  re_acc = re_correct / re_total * 100

  return {
      'accuracy': round(re_acc, 2),
      'label_occurence': re_labels
  }


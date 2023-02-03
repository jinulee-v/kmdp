"""
@module PrunedDependencyRelExtract

Implements RE model that runs GCN on pruned dependency tree.
> Y. Zhang. et al.(2018) Graph Convolution over Pruned Dependency Trees Improves Relation Extraction
"""

import re

import torch
import torch.nn as nn

from . import register_model

from ...victornlp_utils.module import BilinearAttention, GCN

def pruned_depedency_tree(tree, entity1, entity2, prune_k, size):
  """
  Generates bi-directional adjacency matrix of pruned dependency tree
  out of the given dependency tree, span of both entities and the prune distance factor(hyperparameter)

  @param tree List[Dictionary{'dep': X, 'head': X, ...}]. List of dependency arc information.
  @param entity1 List[begin, end]. Token span information.
  @param entity2 List[begin, end]. Token span information.
  @param prune_k Integer. Prune distance. Refer to the original paper for any further definitions.
  @param size Integer. Size of the generated adjacent matrix.
  @return adj_matrix Tensor[size, size]. Symmetric adjacency matrix without an self-loop.
  """
  # tree = [None] + tree
  new_tree = [None] * size
  for arc in tree:
    new_tree[arc['dep']] = arc
  tree = new_tree

  # Find Lowest Common Ancestor(LCA)
  common_ancestors = None 
  for i in list(range(entity1[0], entity1[1] + 1)) + list(range(entity2[0], entity2[1] + 1)):
    ancestors = []
    now = i
    if i >= len(tree):
      print(entity1, entity2)
      print(i, len(tree), size)
      raise ValueError()
    while tree[now] is not None:
      ancestors.append(now)
      assert now == tree[now]['dep']
      # In case of infinity loop,
      if tree[now]['head'] in ancestors:
        break
      now = tree[now]['head']
    ancestors.append(0)

    if common_ancestors is None:
      common_ancestors = ancestors
    else:
      new_common_ancestors = []
      for node in reversed(ancestors):
        if node in common_ancestors:
          new_common_ancestors.append(node)
        else:
          break
      common_ancestors = new_common_ancestors
  lca = common_ancestors[0]

  # Add critical path
  new_tree = [lca]
  for i in list(range(entity1[0], entity1[1] + 1)) + list(range(entity2[0], entity2[1] + 1)):
    if tree[i] is None:
      continue
    assert tree[i]['dep'] == i
    now = i
    while now not in new_tree:
      new_tree.append(now)
      now = tree[now]['head']
  
  # Find auxiliary paths
  NOT_VISITED = 1e6
  distance = torch.ones(size) * NOT_VISITED
  for node in new_tree:
    distance[node] = 0
  for i in range(1, prune_k + 1):
    for arc in tree:
      if arc is None:
        continue
      if arc['dep'] == i - 1 and arc['head'] == NOT_VISITED:
        distance[arc['head']] = i
        new_tree.append(arc['head'])
      if arc['head'] == i - 1 and arc['dep'] == NOT_VISITED:
        distance[arc['dep']] = i
        new_tree.append(arc['dep'])
  
  # Generate adjacency matrix
  adj_matrix = torch.zeros(size, size)
  for node in new_tree:
    if tree[node] is not None and distance[node] <= prune_k:
      head = tree[node]['head']
      dep = tree[node]['dep']
      adj_matrix[dep, head] = 1
      adj_matrix[head, dep] = 1
  return adj_matrix


@register_model('pruned-dependency')
class PrunedDependencyRelExtract(nn.Module):
  """
  @class PrunedDependencyRelExtract

  GCN-over-dependency-tree model for Relation Extraction.
  """

  def __init__(self, embeddings, labels, config):
    """
    Constructor for Deep-Biaffine parser.
    
    @param self The object pointer.
    @param embeddings List of Embedding-like objects. Refer to 'embedding.py' for more details.
    @param labels Dictionary of lists.
    @param config Dictionary config file accessed with 'mtre-sentence' key.
    """
    super(PrunedDependencyRelExtract, self).__init__()

    # Pruning value
    # Infiinity(below 0): No pruning
    # Positive integer >= 0: Pruning
    # - 0: Only the dependency path
    # - 1: Dependency path and its attached values
    self.prune = (config['prune'] >= 0)
    self.prune_k = config['prune'] if self.prune else 1e6

    # Embedding layer
    self.embeddings = nn.ModuleList(embeddings)
    input_size = 0
    for embedding in embeddings:
      input_size += embedding.embed_size
    self.input_size = input_size

    # Bi-LSTM encoder
    self.lstm_hidden_size = config['lstm_hidden_size']
    self.lstm_layers = config['lstm_layers']
    self.encoder = nn.LSTM(input_size=self.input_size, hidden_size=self.lstm_hidden_size, num_layers=self.lstm_layers, batch_first=True, bidirectional=True)

    # - GCN layer
    self.gcn_hidden_size = config['gcn_hidden_size']
    self.gcn_layers = config['gcn_layers']
    self.gcn = GCN(self.lstm_hidden_size * 2, self.gcn_hidden_size, self.gcn_layers)

    # - Prediction Layer
    self.relation_type_size = config['relation_type_size']
    self.re_labels = labels['relation_labels'] # Relation Type Labels
    self.re_labels_stoi = {}
    for i, label in enumerate(self.re_labels):
      self.re_labels_stoi[label] = i
    self.prediction = nn.Sequential(
      nn.Linear(self.gcn_hidden_size * 3, self.relation_type_size),
      nn.ReLU(),
      nn.Linear(self.relation_type_size, len(self.re_labels)),
      nn.LogSoftmax()
    )
  
  def run(self, inputs, **kwargs):
    """
    Runs the model and obtain softmax-ed distribution for each possible labels.
    
    @param self The object pointer.
    @param inputs List of dictionaries. Refer to 'dataset.py' for more details.
    @return relations List, inputs[i]['relations'] flattened. len(relations) == total_relations.
    @return relation_scores Tensor(total_relations, len(relation_labels)). Softmax-ed to obtain label probabilities.
    """
    batch_size = len(inputs)
    device = next(self.parameters()).device

    max_length = -1
    lengths = torch.zeros(batch_size, dtype=torch.long, device=device).detach()
    for i, input in enumerate(inputs):
      length = 1 + sum([len(wp) for wp in input['pos']])
      lengths[i] = length # DP requires [ROOT] node; index 0 is given
      if max_length < length:
        max_length = length
    
    # Embedding
    embedded = []
    for embedding in self.embeddings:
      embedded.append(embedding(inputs))
    embedded = torch.cat(embedded, dim=2)
    
    # Run LSTM encoder first
    lstm_output, _ = self.encoder(embedded)
    duplicated_lstm_output = []

    # Adjacent matrix and entity_embeddings_index
    adj_matrices = []
    entities = []
    # entity_embeddings_index: List[ List[begin, end] -> named_entities ] -> batch
    entity_embeddings_index = []

    # relations
    relations = []
    subj_index = []
    pred_index = []

    for i, input in enumerate(inputs):
      assert 'named_entity' in input and 'pos' in input
      # Find all indices of spaces
      space_indices = [m.start() for m in re.finditer(' ', input['text'])] + [len(input['text'])]
      # Find entity boundaries
      # Specially for KMDP:
      # Indices are found in morpheme IDs, not word-phrase IDs
      entity_embeddings_index.append([])
      for entity in input['named_entity']:
        entities.append(entity)
        entity_embeddings_index[i].append([1])

        assert len(space_indices) == input['word_count']
        # Find word-phrase ID that includes entity boundary
        for m_i, m in enumerate(space_indices):
          # Largest m smaller than 'begin'
          if entity['begin'] > m:
            id = input['pos'][m_i][0]['id']
            for morph in input['pos'][m_i]:
              if entity['text'].startswith(morph['text']):
                id = morph['id']
            entity_embeddings_index[i][-1][0] = id
          # Smallest m larger than 'end'
          if entity['end'] <= m and len(entity_embeddings_index[i][-1]) == 1:
            id = input['pos'][m_i][-1]['id']
            for morph in input['pos'][m_i]:
              if entity['text'].startswith(morph['text']):
                id = morph['id']
            entity_embeddings_index[i][-1].append(id)
        assert len(entity_embeddings_index[i][-1]) == 2

      assert 'dependency' in input and "Input must include golden dependency"
      assert 'relation' in input and "Input must include relation domain(subject and predicate)"

      for rel in input['relation']:
        relations.append(rel)

        duplicated_lstm_output.append(lstm_output[i].unsqueeze(0))
        subj_index.append(entity_embeddings_index[i][rel['subject']])
        pred_index.append(entity_embeddings_index[i][rel['predicate']])
        adj_matrices.append(
          pruned_depedency_tree(
            input['dependency'],
            subj_index[-1],
            pred_index[-1],
            self.prune_k,
            max_length
          ).unsqueeze(0)
        )
    lstm_output = torch.cat(duplicated_lstm_output, dim=0)
    adj_matrices = torch.cat(adj_matrices, dim=0).to(device)

    # Define mask: select tokens only involved in the pruned depedency tree
    mask = (torch.sum(adj_matrices, dim=2) > 0).float()
    mask = mask.unsqueeze(2)

    # GCN-contextualized embeddings
    gcn_output = self.gcn(adj_matrices, lstm_output)
    gcn_output -=  (1 - mask) * 1e6

    # Max-pooling to get sentence vector
    context_embeddings, _ = torch.max(gcn_output, dim=1)
    # context_vector: Tensor(batch, gcn_hidden)

    # Max-pooling function to get entity vector
    def max_pool_entity_vec(batch, begin_end):
      """
      Max-pooling for entities from GCN output.
      @param batch Integer that indicates batch ID
      @param begin_end List of two integers, indicating the begin and last word-phrase of the entity.
      @return entity_embedding Tensor[gcn_hidden_size]. Max-pooled entity embedding
      """
      entity = gcn_output[batch, begin_end[0] : begin_end[1] + 1]
      entity_embedding, _ = torch.max(entity, dim=0)
      return entity_embedding

    # Extract subject and predicate embeddings
    subject_embeddings = []
    predicate_embeddings = []

    for i, relation in enumerate(relations):
      subject_embeddings.append(max_pool_entity_vec(i, subj_index[i]).unsqueeze(0))
      predicate_embeddings.append(max_pool_entity_vec(i, pred_index[i]).unsqueeze(0))

    subject_embeddings = torch.cat(subject_embeddings, dim=0)
    predicate_embeddings = torch.cat(predicate_embeddings, dim=0)

    prediction_input = torch.cat([subject_embeddings, predicate_embeddings, context_embeddings], dim=1)
    
    relation_scores = self.prediction(prediction_input)

    return relations, relation_scores

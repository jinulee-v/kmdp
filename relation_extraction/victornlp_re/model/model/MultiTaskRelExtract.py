"""
@module MultiTaskRelExtract

Implements MTRE model that encorporates dependency parsing and entity type labeling as an auxiliary task.
> Kai, Z. et al.(2019) Multi-Task Learning for Relation Extraction.
"""

import re

import torch
import torch.nn as nn

from . import register_model

from ...victornlp_utils.module import BilinearAttention, GCN

@register_model('mtre-sentence')
class MultiTaskRelExtract_Sentence(nn.Module):
  """
  @class MultiTaskRelExtract

  Multi-task model for Relation Extraction.
  Encorporates DP and entity type labeling as an auxiliary task.

  NOTE that 'bag' concept is not necessarily present in every relation-extraction datasets.
  We do not implement bag-wise attention, but only intra-sentence attention with MLP.

  For Entity Type Labeling,
  - This model tries to classify every named entity provided within the sentence.
  - This model implements multi-class classification, regularized with mean/standard deviation.

  For Relation Extractor,
  - This model exploits predicate/subject's embedding and the attentive context vector.
  """

  def __init__(self, embeddings, labels, config):
    """
    Constructor for Deep-Biaffine parser.
    
    @param self The object pointer.
    @param embeddings List of Embedding-like objects. Refer to 'embedding.py' for more details.
    @param labels Dictionary of lists.
    @param config Dictionary config file accessed with 'mtre-sentence' key.
    """
    super(MultiTaskRelExtract_Sentence, self).__init__()

    # Embedding layer
    self.embeddings = nn.ModuleList(embeddings)
    input_size = 0
    for embedding in embeddings:
      input_size += embedding.embed_size
    self.input_size = input_size

    # Bi-GRU encoder
    self.gru_hidden_size = config['gru_hidden_size']
    self.gru_layers = config['gru_layers']
    self.encoder = nn.GRU(input_size=self.input_size, hidden_size=self.gru_hidden_size, num_layers=self.gru_layers, batch_first=True, bidirectional=True)
    
    # Deep-Biaffine parser subunit
    self.dp_arc_size = config['dependency_arc_size']
    self.dp_dropout = config['dependency_dropout']
    self.W_head = nn.Sequential(
      nn.Linear(self.gru_hidden_size * 2, self.dp_arc_size),
      nn.ReLU(),
      nn.Dropout(self.dp_dropout)
    )
    self.W_dep = nn.Sequential(
      nn.Linear(self.gru_hidden_size * 2, self.dp_arc_size),
      nn.ReLU(),
      nn.Dropout(self.dp_dropout)
    )
    self.dependency_attention = BilinearAttention(self.dp_arc_size, self.dp_arc_size, 1)

    # Entity Type Classifier &
    # Relation Extractor
    """
    NOTE: Different from original paper.
    Refer to the docstring of this class for better description.
    """
    # - Entity Type Classifier
    self.entity_type_size = config['entity_type_size']
    self.etl_labels = labels['entity_type_labels'] # Entity Type Labels
    self.etl_labels_stoi = {}
    for i, label in enumerate(self.etl_labels):
      self.etl_labels_stoi[label] = i
    self.classifier = nn.Sequential(
      nn.Linear(self.gru_hidden_size * 2, self.entity_type_size),
      nn.ReLU(),
      nn.Linear(self.entity_type_size, len(self.etl_labels)),
      nn.Sigmoid()
    )

    # - GCN layer
    self.gcn_hidden_size = config['gcn_hidden_size']
    self.gcn_layers = config['gcn_layers']
    self.gcn = GCN(self.gru_hidden_size * 2, self.gcn_hidden_size, self.gcn_layers)
    self.symmetric_adj_matrix = bool(config['symmetric_adj_matrix']) if 'symmetric_adj_matrix' in config else False

    # - Attentive Context Vector
    self.W_tok = nn.Linear(self.gcn_hidden_size, self.gcn_hidden_size)
    self.q_query = nn.Parameter(torch.randn(1, 1, self.gcn_hidden_size))

    # - Prediction Layer
    self.relation_type_size = config['relation_type_size']
    self.re_labels = labels['relation_labels'] # Relation Type Labels
    self.re_labels_stoi = {}
    for i, label in enumerate(self.re_labels):
      self.re_labels_stoi[label] = i
    self.prediction = nn.Sequential(
      # nn.Linear(self.gcn_hidden_size * 3, self.relation_type_size),
      nn.Linear(self.gcn_hidden_size, self.relation_type_size),
      nn.ReLU(),
      nn.Linear(self.relation_type_size, len(self.re_labels)),
      nn.LogSoftmax(1)
    )
  
  def run(self, inputs, pretrain=False, lengths=None, mask=None):
    """
    Runs the model and obtain softmax-ed distribution for each possible labels.
    
    @param self The object pointer.
    @param inputs List of dictionaries. Refer to 'dataset.py' for more details.
    @param pretrain If True, run the relation extraction module. If False, skip thos phases.
    @return relations List, inputs[i]['relations'] flattened. len(relations) == total_relations.
    @return relation_scores Tensor(total_relations, len(relation_labels)). Softmax-ed to obtain label probabilities.
    @return arc_attention Tensor(batch, labels). scores[i][j] contains the log(probability) of i-th sentence in batch having j-th label.
    @return entities List, inputs[i]['named_entities'] flattened. len(entities) == total_entities.
    @return entity_type_scores Tensor(total_entities, len(type_labels)). Not softmaxed, but normalized to E(X)=0 / stddev(X)=1 for multi-class classification.
    """
    batch_size = len(inputs)
    if lengths is None:
      lengths = torch.zeros(batch_size, dtype=torch.long).detach()
      for i, input in enumerate(inputs):
        lengths[i] = input['word_count'] + 1 # DP requires [ROOT] node; index 0 is given
    else:
      assert lengths.dim()==1 and lengths.size(0) == batch_size
      lengths = lengths.detach()
    
    # Embedding
    embedded = []
    for embedding in self.embeddings:
      embedded.append(embedding(inputs))
    embedded = torch.cat(embedded, dim=2)
    
    # Run GRU encoder first
    gru_output, _ = self.encoder(embedded)

    # Dependency parser
    head_vectors = self.W_head(gru_output)
    dep_vectors = self.W_dep(gru_output)
    arc_attention = self.dependency_attention(head_vectors, dep_vectors, lengths)
    if mask is None:
      mask = torch.zeros_like(arc_attention).detach()
      for i, length in enumerate(lengths):
        for j in range(1, length):
          mask[i, 0, j, :length] = 1
    else:
      assert mask.size()==arc_attention.size()
      mask = mask.detach()
  
    arc_attention = arc_attention - ((1-mask) * 100000)
    arc_attention = nn.functional.log_softmax(arc_attention, 3)

    # Entity Type Labeling
    entities = []
    entity_embeddings = []
    entity_embeddings_index = []
    for i, input in enumerate(inputs):
      # Find entity boundaries
      for entity in input['named_entity']:
        entities.append(entity)
        entity_embeddings_index.append([])

        # Find all indices of spaces
        space_indices = [m.start() for m in re.finditer(' ', input['text'])] + [len(input['text'])]
        assert len(space_indices) == input['word_count']
        for m_i, m in enumerate(space_indices):
          if entity['end'] <= m:
            last_morph_id = input['pos'][m_i][-1]['id']
            entity_embeddings.append(gru_output[i, last_morph_id].unsqueeze(0)) # +1 is for extra ROOT at the beginning
            entity_embeddings_index[i].append(last_morph_id)
    entity_embeddings = torch.cat(entity_embeddings, dim=0)
    # entity_embeddings: Tensor(total_entities, 2*gru_hidden_size)
    entity_type_scores = self.classifier(entity_embeddings)
    # entity_type_scores: Tensor(total_entities, len(entity_type_labels))
    entity_type_scores = (entity_type_scores - torch.mean(entity_type_scores, dim=1, keepdim=True)) / torch.std(entity_type_scores, dim=1, keepdim=True)

    # Relation Extraction
    if not pretrain:
      relations = []
      subject_embeddings = []
      predicate_embeddings = []
      context_embeddings = []

      arc_attention_temp = arc_attention.squeeze(1).transpose(1, 2)
      if self.symmetric_adj_matrix:
        arc_attention_temp = (arc_attention_temp + arc_attention_temp) / 2

      # GCN-contextualized embeddings
      gcn_output = self.gcn(torch.exp(arc_attention_temp), gru_output)

      # Token-wise contextualized embeddings
      token_wise_attention = torch.sum(self.W_tok(gcn_output) * self.q_query, dim=2, keepdim=True)
      # token_wise_attention: Tensor(batch, length, 1)
      mask_attention = mask[:, 0, :, 0].unsqueeze(2)
      # mask_attention: Tensor(batch, length, 1) ; effects of the dummy ROOT is removed
      context_vector = torch.sum(gcn_output * torch.softmax(token_wise_attention, dim=1) * mask_attention, dim=1)
      # context_vector: Tensor(batch, gcn_hidden)

      # Prediction
      for i, input in enumerate(inputs):
        for relation in input['relation']:
          relations.append(relation)
          
          # subject_embeddings.append(gcn_output[i, entity_embeddings_index[i][relation['subject']]].unsqueeze(0))
          # predicate_embeddings.append(gcn_output[i, entity_embeddings_index[i][relation['predicate']]].unsqueeze(0))
          context_embeddings.append(context_vector[i].unsqueeze(0))

      # subject_embeddings = torch.cat(subject_embeddings, dim=0)
      # predicate_embeddings = torch.cat(predicate_embeddings, dim=0)
      context_embeddings = torch.cat(context_embeddings, dim=0)

      # prediction_input = torch.cat([subject_embeddings, predicate_embeddings, context_embeddings], dim=1)
      prediction_input = context_embeddings

      relation_scores = self.prediction(prediction_input)
    else:
      relations = None
      relation_scores = None
        

    return relations, relation_scores, arc_attention, entities, entity_type_scores

"""
@module bert_embedding
Various transformer-based embedding modules.

Class name format: 
  Embedding(description)_language
Requirements:
  __init__(self, config)
    @param config Dictionary. Configuration for Embedding*_* (free-format)
  self.embed_size
  forward(self, inputs)
    @param inputs List of dictionaries. Refer to 'corpus.py' for more details.
    @return output Tensor(batch, length+1, self.embed_size). length+1 refers to the virtual root.
"""
import itertools

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

from . import register_embedding
from .KorBERT_morph_tokenizer import BertTokenizer as KorBertTokenizer

lexical_morphemes_tag = [
  'NNG', 'NNP', 'NNB', 'NR', 'NP',
  'VV', 'VA', 'VX', 'VCP', 'VCN',
  'MMA', 'MMD', 'MMN' 'MAG', 'MAJ',
  'IC', 'SL', 'SN', 'SH', 'XR', 'NF', 'NV'
]

replace = {
  '/MMA': '/MM',
  '/MMD': '/MM',
  '/MMN': '/MM',
  '/NV': '/NA',
  '/NF': '/NA',
  '/XR': '/NNG'
}

@register_embedding('etri-korbert')
class EmbeddingBERTMorph_kor(nn.Module):
  """
  Morpheme level Embedding model using KorBERT(ETRI, 2020).
  Initially outputs hidden states for every morphemes,
  but can configurate to select word-phrase level embeddings.
  """
  def __init__(self, config):
    """
    Constructor for EmbeddingBERTMorph_kor.
    
    @param self The object pointer.
    @param config Dictionary. Configuration for EmbeddingBERTMorph_kor
    """
    super(EmbeddingBERTMorph_kor, self).__init__()

    # Load model from configuration
    model_dir = config['file_directory']
    self.tokenizer = KorBertTokenizer.from_pretrained(model_dir)
    self.model = BertModel.from_pretrained(model_dir)
    self.fine_tune = bool(config['train'])
    if self.fine_tune:
      self.model.train()
      self.model.requires_grad = True
    else:
      self.model.eval()
      self.model.requires_grad = False

    self.is_word_phrase_embedding = config['word_phrase']
    self.embed_size = 1536 if self.is_word_phrase_embedding else 768
    self.special_tokens = config['special_tokens']
  
  def forward(self, inputs):
    """
    Overridden forward().
    
    @param self The object pointer.
    @param inputs List of dictionary. Refer to 'corpus.py' for more details.
    
    @return Tensor(batch, length, self.embed_size)
    """
    device = next(self.parameters()).device
    
    tokens_list = []
    sentences = []
    attention_mask = []
    lengths = torch.zeros(len(inputs), dtype=torch.long)
    # List for word phrase/morpheme recovery index
    if self.is_word_phrase_embedding:
      selects = [([[0, 0]] if 'bos' in self.special_tokens else []) for _ in range(len(inputs))]
    else:
      selects = [([0] if 'bos' in self.special_tokens else []) for _ in range(len(inputs))]

    for i, input in enumerate(inputs):
      # Input format:
      # ETRI/SL 에서/JKB 한국어/NNP BERT/SL 언어/NNG 모델/NNG 을/JKO 배포/NNG 하/XSV 었/EP 다/EF ./SF
      wp_last_tokens = []
      tokens = ['[CLS]', '_']
      j = 1 if 'bos' in self.special_tokens else 0
      for word_phrase in input['pos']:
        for morph in word_phrase:
          joined = morph['text']+'/'+morph['pos_tag']
          # Replace incompatible PoS tags between Sejong and ETRI format
          for sejong, etri in replace.items():
            joined = joined.replace(sejong, etri)
          tokenized = self.tokenizer.tokenize(joined)
          j += len(tokenized)
          tokens.extend(tokenized)
        wp_last_tokens.append(j)
      assert len(wp_last_tokens) == input['word_count']
      tokens.extend(['[SEP]', '_'])

      tokens_list.append(tokens)
      token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
      sentences.append(torch.tensor(token_ids, dtype=torch.long))
      attention_mask.append(torch.ones([len(tokens)], dtype=torch.long))
      lengths[i] = len(tokens) - 2 # index of [SEP]

      if self.is_word_phrase_embedding:
        # Word phrase recovery: select last lexical / functional morpheme from each WP
        pos_tag = lambda token: token.split('/')[-1].replace('_', '')
        new_pair = [None, None] # lexical & functinoal index
        pos_index = 0
        for j, token in enumerate(tokens):
          if pos_index >= len(input['pos']):
            break
          if token.endswith('_') and len(token)>1:
            if pos_tag(token) in lexical_morphemes_tag:
              new_pair[0] = j
            if pos_tag(token) not in lexical_morphemes_tag:
              new_pair[1] = j
          # Last morpheme in word phrase
          if j == wp_last_tokens[pos_index]:
            selects[i].append(new_pair)
            new_pair = [None, None] 
            pos_index += 1
      else:
        # Morpheme recovery: select PoS tag(which contains contextual lexical information)
        for j, token in enumerate(tokens):
          if token == '[UNK]':
            selects[i].append(j)
          elif token.endswith('_') and len(token) > 1:
            selects[i].append(j)
        assert len(selects[i]) == input['pos'][-1][-1]['id'] + 1

    sentences = nn.utils.rnn.pad_sequence(sentences, batch_first=True).to(device)
    attention_mask = nn.utils.rnn.pad_sequence(attention_mask, batch_first=True).to(device)
    # eos token treatment for word phrase embedding
    if 'eos' in self.special_tokens:
      for length, select in zip(lengths, selects):
        if self.is_word_phrase_embedding:
          select.append([length, length])
        else:
          select.append(length)
          
    # Run BERT model
    if self.fine_tune:
      output = self.model(sentences, attention_mask, torch.zeros_like(attention_mask))['last_hidden_state']
    else:
      with torch.no_grad():
        output = self.model(sentences, attention_mask, torch.zeros_like(attention_mask))['last_hidden_state']

    # Use `selects` to:
    # Return word-phrase embeddings
    if self.is_word_phrase_embedding:
      embedded = []
      for sent, select in zip(output, selects):
        embedded.append(torch.zeros(len(select), self.embed_size).to(device))
        for i, pair in enumerate(select):
          left = pair[0]; right = pair[1]
          if left:
            embedded[-1][i][:self.embed_size//2] = sent[left]
          if right:
            embedded[-1][i][self.embed_size//2:] = sent[right]
      embedded = nn.utils.rnn.pad_sequence(embedded, batch_first=True)
      return embedded

    # Return embeddings for all morphemes
    else:
      embedded = []
      for sent, select in zip(output, selects):
        embedded.append(sent[select])
      embedded = nn.utils.rnn.pad_sequence(embedded, batch_first=True)
      return embedded

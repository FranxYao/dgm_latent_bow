"""The dataset class

Yao Fu, Columbia University 
yao.fu@columbia.edu
Tue 30th Jul 2019 
"""

import wikibio
import copy 

import numpy as np 

from wikibio.DataLoader import DataLoader
from wikibio.preprocess import Vocab
from tqdm import tqdm
from nltk.corpus import stopwords

def _get_sent_len(s, end_id):
  for i, si in enumerate(s):
    if(si == end_id): break
  return i

def _build_batch(sents, vals, keys, pos, 
  max_enc_len, max_dec_len, max_bow_len, start_id, end_id, pad_id):
  """Build a batch data"""
  def _pad(s, maxlen):
    s_ = copy.deepcopy(s)
    s_ = s_[: maxlen]
    slen = len(s_)
    if(slen < maxlen): s_ += [pad_id] * (maxlen - slen)
    return s_

  enc_keys, enc_locs, enc_vals, enc_lens = [], [], [], []
  dec_bow, dec_bow_len = [], []
  dec_inputs, dec_targets, dec_lens = [], [], []
  batch_size = len(sents)
  for i in range(batch_size):
    s, v, k, p = sents[i], vals[i], keys[i], pos[i]
  # for s, v, k, p in zip(sents, vals, keys, pos):
    elen = len(v)
    enc_lens.append(elen)
    k_ = _pad(k, max_enc_len)
    enc_keys.append(k_)
    p_ = _pad(p, max_enc_len)
    enc_locs.append(p_)
    v_ = _pad(v, max_enc_len)
    enc_vals.append(v_)
    
    dbow = list(set(s))
    dbow_len = len(dbow)
    dbow = _pad(dbow, max_bow_len)
    dec_bow.append(dbow)
    dec_bow_len.append(dbow_len)

    slen = len(s) + 1
    s = [start_id] + s + [end_id]
    s_in = _pad(s[:-1], max_dec_len)
    s_out = _pad(s[1:], max_dec_len)
    dec_inputs.append(s_in)
    dec_targets.append(s_out)
    dec_lens.append(slen)

  batch = { 'enc_keys': np.array(enc_keys) ,
            'enc_locs': np.array(enc_locs),
            'enc_vals': np.array(enc_vals),
            'enc_lens': np.array(enc_lens),
            'dec_bow': np.array(dec_bow),
            'dec_bow_len': np.array(dec_bow_len),
            'dec_inputs': np.array(dec_inputs),
            'dec_targets': np.array(dec_targets),
            'dec_lens': np.array(dec_lens),
            'references': sents }
  return batch

class DatasetTable2text(object):
  def __init__(self, config):
    self.start_id = config.dec_start_id
    self.end_id = config.dec_end_id
    self.pad_id = config.pad_id

    self.dir = config.data2text_dir
    self.limits = config.data2text_limits
    self.word_vocab_path = config.data2text_word_vocab_path
    self.field_vocab_path = config.data2text_field_vocab_path

    self.max_enc_len = config.max_enc_len['wikibio']
    self.max_dec_len = config.max_dec_len['wikibio']
    self.max_bow_len = config.max_dec_bow

    self.stop_words = None

    self.word2id = None
    self.id2word = None
    self.key2id = None
    self.id2key = None

    self._dataset = {'train': None, 'dev': None, 'test': None}
    self._ptr = {'train': 0, 'dev': 0, 'test': 0}
    return 

  @property
  def vocab_size(self): return len(self.word2id)

  def dataset_size(self, setname):
    return len(self._dataset[setname][0])

  def num_batches_dynamic(self, batch_size, setname):
    nb = self.dataset_size(setname) // batch_size
    if(self.dataset_size(setname) % batch_size == 0): return nb
    else: return nb + 1

  def num_batches(self, batch_size, setname):
    return self._dataset[setname][0].shape[0]

  def build(self):
    """Build the wikibio dataset class"""
    print('Building the wikibio dataset ... ')
    dataloader = DataLoader(self.dir, self.limits)
    vocab = Vocab(self.word_vocab_path, self.field_vocab_path)

    self.word2id = vocab.word2id
    self.id2word = vocab.id2word
    self.key2id = vocab.key2id
    self.id2key = vocab.id2key
    

    self._dataset['train'] = dataloader.train_set 
    self._dataset['dev'] = dataloader.dev_set
    self._dataset['test'] = dataloader.test_set
    return 

  def save(self):
    def _save_set(setname):
      num_batches = self.num_batches(100, setname)
      enc_keys, enc_locs, enc_vals, enc_lens = [], [], [], []
      dec_bow, dec_bow_len = [], []
      dec_inputs, dec_targets, dec_lens = [], [], []
      for _ in tqdm(range(num_batches - 1)):
        batch = self.next_batch_from_sent(setname, 100)
        enc_keys.append(batch['enc_keys'])
        enc_locs.append(batch['enc_locs'])
        enc_vals.append(batch['enc_vals'])
        enc_lens.append(batch['enc_lens'])
        dec_bow.append(batch['dec_bow'])
        dec_bow_len.append(batch['dec_bow_len'])
        dec_inputs.append(batch['dec_inputs'])
        dec_targets.append(batch['dec_targets'])
        dec_lens.append(batch['dec_lens'])

      enc_keys = np.stack(enc_keys)
      enc_locs = np.stack(enc_locs)
      enc_vals = np.stack(enc_vals)
      enc_lens = np.stack(enc_lens)
      dec_bow = np.stack(dec_bow)
      dec_bow_len = np.stack(dec_bow_len)
      dec_inputs = np.stack(dec_inputs)
      dec_targets = np.stack(dec_targets)
      dec_lens = np.stack(dec_lens)
      np.save('../data/wikibio/%s_enc_keys' % setname, enc_keys)
      np.save('../data/wikibio/%s_enc_locs' % setname, enc_locs)
      np.save('../data/wikibio/%s_enc_vals' % setname, enc_vals)
      np.save('../data/wikibio/%s_enc_lens' % setname, enc_lens)
      np.save('../data/wikibio/%s_dec_bow' % setname, dec_bow)
      np.save('../data/wikibio/%s_dec_bow_len' % setname, dec_bow_len)
      np.save('../data/wikibio/%s_dec_inputs' % setname, dec_inputs)
      np.save('../data/wikibio/%s_dec_targets' % setname, dec_targets)
      np.save('../data/wikibio/%s_dec_lens' % setname, dec_lens)
      return 
    
    _save_set('train')
    _save_set('dev')
    return 

  def load(self):
    vocab = Vocab(self.word_vocab_path, self.field_vocab_path)
    self.word2id = vocab.word2id
    self.id2word = vocab.id2word
    self.key2id = vocab.key2id
    self.id2key = vocab.id2key
    self.stop_words = set(stopwords.words('english'))
  
    def _load_set(setname):
      enc_keys = np.load('../data/wikibio/%s_enc_keys.npy' % setname)
      enc_locs = np.load('../data/wikibio/%s_enc_locs.npy' % setname)
      enc_vals = np.load('../data/wikibio/%s_enc_vals.npy' % setname)
      enc_lens = np.load('../data/wikibio/%s_enc_lens.npy' % setname)
      dec_bow = np.load('../data/wikibio/%s_dec_bow.npy' % setname)
      dec_bow_len = np.load('../data/wikibio/%s_dec_bow_len.npy' % setname)
      dec_inputs = np.load('../data/wikibio/%s_dec_inputs.npy' % setname)
      dec_targets = np.load('../data/wikibio/%s_dec_targets.npy' % setname)
      dec_lens = np.load('../data/wikibio/%s_dec_lens.npy' % setname)
      return (enc_keys, enc_locs, enc_vals, enc_lens, dec_bow, dec_bow_len, 
        dec_inputs, dec_targets, dec_lens)
    for setname in ['train', 'dev']:
      data = _load_set(setname)
      self._dataset[setname] = data
    return 

  def next_batch(self, setname, batch_size=None, model_name=None):
    ptr = self._ptr[setname]
    num_batches = self._dataset[setname][0].shape[0]
    references = []
    for d, l in zip(self._dataset[setname][7][ptr], 
      self._dataset[setname][8][ptr]):
      references.append([d[: l]])
    batch = { 'enc_keys':    self._dataset[setname][0][ptr],
              'enc_locs':    self._dataset[setname][1][ptr],
              'enc_vals':    self._dataset[setname][2][ptr], 
              'enc_lens':    self._dataset[setname][3][ptr], 
              'dec_bow':     self._dataset[setname][4][ptr], 
              'dec_bow_len': self._dataset[setname][5][ptr], 
              'dec_inputs':  self._dataset[setname][6][ptr], 
              'dec_targets': self._dataset[setname][7][ptr], 
              'dec_lens':    self._dataset[setname][8][ptr],
              'references':  references}
    self._ptr[setname] = (self._ptr[setname] + 1) % num_batches 
    return batch

  def next_batch_from_sent(self, setname, batch_size, model_name=None):
    ptr = self._ptr[setname]
    sents_ = self._dataset[setname][0][ptr: ptr + batch_size]
    vals_ = self._dataset[setname][1][ptr: ptr + batch_size]
    keys_ = self._dataset[setname][2][ptr: ptr + batch_size]
    pos_ = self._dataset[setname][3][ptr: ptr + batch_size]
    sents, vals, keys, pos, _ = self._dataset[setname]
    sents_ = sents[ptr: ptr + batch_size]
    vals_ = vals[ptr: ptr + batch_size]
    keys_ = keys[ptr: ptr + batch_size]
    pos_ = pos[ptr: ptr + batch_size]
    batch = _build_batch(
      sents_, vals_, keys_, pos_, 
      self.max_enc_len, self.max_dec_len, self.max_bow_len,
      self.start_id, self.end_id, self.pad_id)
    
    ptr = ptr + batch_size
    if(ptr >= self.dataset_size(setname)): self._ptr[setname] = 0
    else: self._ptr[setname] = ptr
    return batch

  def print_predict(self, model_name,
    output_dict=None, batch_dict=None, output_fd=None, num_cases=0):
    """print the output prediction"""
    batch_size = batch_dict['enc_keys'].shape[0]
    for i in range(batch_size):
      str_out  = 'enc_keys    | '
      elen = batch_dict['enc_lens'][i]
      str_out += ' '.join(
        [self.id2key[k] for k in batch_dict['enc_keys'][i][: elen]]) + '\n'
      str_out += 'enc_locs    | '
      str_out += ' '.join(
        [str(l) for l in batch_dict['enc_locs'][i][: elen]]) + '\n'
      str_out += 'enc_vals    | '
      str_out += ' '.join(
        [self.id2word[w] for w in batch_dict['enc_vals'][i][: elen]]) + '\n'

      dblen = batch_dict['dec_bow_len'][i]
      str_out += 'dec_bow     | '
      str_out += ' '.join(
        [self.id2word[w] for w in batch_dict['dec_bow'][i][: dblen]]) + '\n'

      # dlen = batch_dict['dec_lens'][i]
      # str_out += 'dec_inputs: '
      # str_out += ' '.join(
      #   [self.id2word[w] for w in batch_dict['dec_inputs'][i][: dlen]]) + '\n'
      # str_out += 'dec_targets: '
      # str_out += ' '.join(
      #   [self.id2word[w] for w in batch_dict['dec_targets'][i][: dlen]]) + '\n'

      str_out += 'reference   | '
      str_out += ' '.join(
        [self.id2word[w] for w in batch_dict['references'][i][0]]) + '\n'
      
      if(output_dict is not None):
        str_out += 'dec_predict | '
        olen = _get_sent_len(output_dict['dec_predict'][i], self.end_id)
        str_out += ' '.join(
          [self.id2word[w] for w in output_dict['dec_predict'][i][: olen]]) + '\n'

      if(i < num_cases): print(str_out)
    return 
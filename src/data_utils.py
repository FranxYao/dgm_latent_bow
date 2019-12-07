"""Data utility functions

Current datasets: 
* MSCOCO (Lin et. al. 2014)] (http://cocodataset.org/)
* Quora (https://www.kaggle.com/c/quora-question-pairs)
  In this dataset, originally, the training set is 60M while the test set is 
  299M (for classification purpose). This is not a typical generation setting. 
  So the inverse the training and the testing set
* Twitter URL (Lan et. al. 2017)
* PPDB: The Paraphrase Database (Ganitkevitch et. al. 2013)
* PIT-2015: Twitter paraphrase Corpus (Xu et. al. 2014, 2015)
* MSRP: MSR Paraphrase Corpus (Dolan et. al. 2004) 
  This dataset only contains 5800 sentence pairs, too small for generation, 
  abandon 

This time we try spacy for data processing (https://spacy.io/)

Yao Fu, Columbia University 
yao.fu@columabi.edu
Mar 05TH 2019
"""

import nltk 
import json

import numpy as np 

from collections import Counter
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from nlp_pipeline import *


def quora_read(file_path, bleu_baseline=False):
  """Read the quora dataset"""
  print("Reading quora raw data .. ")
  print("  data path: %s" % file_path)
  with open(file_path) as fd:
    lines = fd.readlines()
  sentence_sets = []
  for l in tqdm(lines):
    p0, p1 = l[:-1].lower().split("\t")
    sentence_sets.append([word_tokenize(p0), word_tokenize(p1)])

  if(bleu_baseline):
    print("calculating bleu ... ")
    hypothesis = [s[0] for s in sentence_sets]
    references = [s[1:] for s in sentence_sets]
    bleu = corpus_bleu(references, hypothesis)
    print("bleu on the training set: %.4f" % bleu)
  return sentence_sets

def mscoco_read_json(file_path, bleu_baseline=False):
  """Read the mscoco dataset

  Args:
    file_path: path to the raw data, a string

  Returns:
    sentence_sets: the sentence sets, a list of paraphrase lists
  """
  print("Reading mscoco raw data .. ")
  print("  data path: %s" % file_path)
  with open(file_path, "r") as fd:
    data = json.load(fd)

  print("%d sentences in total" % len(data["annotations"]))
  
  # aggregate all sentences of the same images
  image_idx = set([d["image_id"] for d in data["annotations"]])
  paraphrases = {}
  for im in image_idx: paraphrases[im] = []
  for d in tqdm(data["annotations"]):
    im = d["image_id"]
    sent = d["caption"]
    paraphrases[im].append(word_tokenize(sent))

  sentence_sets = [paraphrases[im] for im in paraphrases]

  # blue on the training set, a baseline/ upperbound
  if(bleu_baseline):
    print("calculating bleu ... ")
    hypothesis = [s[0] for s in sentence_sets]
    references = [s[1:] for s in sentence_sets]
    bleu = corpus_bleu(references, hypothesis)
    print("bleu on the training set: %.4f" % bleu)
  return sentence_sets

def build_batch_seq2seq_bow2seq(data_batch, 
                                len_batch, 
                                stop_words, 
                                max_enc_bow, 
                                max_dec_bow,
                                pad_id):
  """First predict all target sequence, then bow to sequence"""
  enc_inputs = []
  enc_lens = []
  enc_targets = []
  enc_seq2seq_inputs = []
  enc_seq2seq_targets = []
  enc_seq2seq_lens = []

  dec_bow = []
  dec_bow_len = []
  dec_inputs = []
  dec_targets = []
  dec_lens = [] 

  def _pad(s_set, max_len, pad_id):
    s_set = list(s_set)[: max_len]
    for i in range(max_len - len(s_set)): s_set.append(pad_id)
    return s_set

  for st, slen in zip(data_batch, len_batch):
    para_bow = set()
    for s in st: para_bow |= set(s)
    para_bow -= stop_words
    para_bow = _pad(para_bow, max_enc_bow, pad_id)

    # for i in range(5):
    i = 0
    j = (i + 1) % 5
    inp = st[i][: -1]
    d_in = st[j][: -1]
    d_out = st[j][1: ]
    len_inp = slen[i]
    len_out = slen[j]

    enc_inputs.append(inp)
    enc_lens.append(len_inp)

    enc_s2s_inp = []
    enc_s2s_tgt = []
    enc_s2s_len = []
    for k in range(5):
      if(k == i): continue
      enc_s2s_inp.append(st[k][: -1])
      enc_s2s_tgt.append(st[k][1: ])
      enc_s2s_len.append(slen[k])

    enc_seq2seq_inputs.append(enc_s2s_inp)
    enc_seq2seq_targets.append(enc_s2s_tgt)
    enc_seq2seq_lens.append(enc_s2s_len)

    d_bow = set(d_in) - stop_words
    d_bow_len = len(d_bow)
    d_bow_ = d_bow
    d_bow = _pad(d_bow, max_dec_bow, pad_id)

    e_bow = []
    for k in range(5):
      if(k == i): continue
      e_bow.extend(st[k][: -1])
    e_bow = set(e_bow) - stop_words
    e_bow -= set(inp)
    e_bow = _pad(e_bow, max_enc_bow, pad_id)
    enc_targets.append(e_bow)

    # enc_targets.append(para_bow)

    dec_bow.append(d_bow)
    dec_bow_len.append(d_bow_len)
    dec_inputs.append(d_in)
    dec_targets.append(d_out)
    dec_lens.append(len_out)

  batch_dict = {"enc_inputs":           np.array(enc_inputs),
                "enc_lens":             np.array(enc_lens),
                "enc_seq2seq_inputs":   np.array(enc_seq2seq_inputs),
                "enc_seq2seq_targets":  np.array(enc_seq2seq_targets),
                "enc_seq2seq_lens":     np.array(enc_seq2seq_lens), 
                "enc_targets":          np.array(enc_targets),
                "references":           enc_seq2seq_targets, 
                "dec_bow":              np.array(dec_bow),
                "dec_bow_len":          np.array(dec_bow_len),
                "dec_inputs":           np.array(dec_inputs),
                "dec_targets":          np.array(dec_targets),
                "dec_lens":             np.array(dec_lens)}
  return batch_dict

def build_batch_bow_seq2seq_eval(
  data_batch, len_batch, stop_words, max_enc_bow, max_dec_bow, pad_id,
  single_ref=False):
  """Build evaluation batch, basically the same as the seq2seq setting"""
  enc_inputs = []
  enc_lens = []
  references = []
  ref_lens = [] 
  dec_golden_bow = []
  dec_bow = []
  dec_bow_len = []

  def _pad(s_set, max_len, pad_id):
    s_set = list(s_set)[: max_len]
    for i in range(max_len - len(s_set)): s_set.append(pad_id)
    return s_set

  def _pad_golden(s_set, max_len):
    s_set_ = list(s_set)
    s_set = list(s_set)[: max_len]
    for i in range(max_len - len(s_set)): 
      s_set.append(np.random.choice(s_set_))
    return s_set

  for st, slen in zip(data_batch, len_batch):
    inp = st[0][: -1]
    len_inp = slen[0]
    if(single_ref): 
      ref = [s_[1: s_len_] for s_, s_len_ in zip(st[-1:], slen[-1:])]
    else: ref = [s_[1: s_len_] for s_, s_len_ in zip(st[1:], slen[1:])]

    golden_bow = []
    for r in ref: golden_bow.extend(r)
    golden_bow = set(golden_bow) - stop_words 
    golden_bow = _pad_golden(golden_bow, max_enc_bow)

    d_in = st[1][: -1]
    d_bow = set(d_in) - stop_words
    d_bow_len = len(d_bow)
    d_bow_ = d_bow
    d_bow = _pad(d_bow, max_dec_bow, pad_id)
    dec_bow.append(d_bow)
    dec_bow_len.append(d_bow_len)

    enc_inputs.append(inp)
    enc_lens.append(len_inp)
    references.append(ref)
    dec_golden_bow.append(golden_bow)
  
  batch_dict = {"enc_inputs": np.array(enc_inputs),
                "enc_lens": np.array(enc_lens),
                "golden_bow": np.array(dec_golden_bow),
                "dec_bow": np.array(dec_bow),
                "dec_bow_len": np.array(dec_bow_len),
                "references": references}
  return batch_dict

def build_batch_bow_seq2seq(data_batch, 
                            len_batch, 
                            stop_words, 
                            max_enc_bow, 
                            max_dec_bow,
                            pad_id,
                            source_bow=True):
  """Build a training batch for the bow seq2seq model"""
  enc_inputs = []
  enc_lens = []
  enc_targets = []
  dec_bow = []
  dec_bow_len = []
  dec_inputs = []
  dec_targets = []
  dec_lens = [] 

  def _pad(s_set, max_len, pad_id):
    s_set = list(s_set)[: max_len]
    for i in range(max_len - len(s_set)): s_set.append(pad_id)
    return s_set

  for st, slen in zip(data_batch, len_batch):
    para_bow = set()
    for s in st: para_bow |= set(s)
    para_bow -= stop_words
    para_bow = _pad(para_bow, max_enc_bow, pad_id)

    num_para = len(st)

    for i in range(num_para):
      j = (i + 1) % num_para
      inp = st[i][: -1]
      d_in = st[j][: -1]
      d_out = st[j][1: ]
      len_inp = slen[i]
      len_out = slen[j]

      enc_inputs.append(inp)
      enc_lens.append(len_inp)

      d_bow = set(d_in) - stop_words
      d_bow_len = len(d_bow)
      d_bow_ = d_bow
      d_bow = _pad(d_bow, max_dec_bow, pad_id)

      e_bow = []
      for k in range(num_para):
        if(k == i): 
          continue
          # if(source_bow == False):
          #   continue
        e_bow.extend(st[k][: -1])
      e_bow = set(e_bow) - stop_words

      # method 1: pad bow
      # do not predict source words
      if(source_bow == False):
        e_bow -= set(inp) 
      e_bow = _pad(e_bow, max_enc_bow, pad_id)

      # method 2: repeat bow
      # e_bow = list(e_bow)
      # e_bow_ = []
      # i = 0
      # while(len(e_bow_) < max_enc_bow):
      #   e_bow_.append(e_bow[i])
      #   i = (i + 1) % len(e_bow)
      # e_bow = e_bow_

      enc_targets.append(e_bow)

      # enc_targets.append(para_bow)

      dec_bow.append(d_bow)
      dec_bow_len.append(d_bow_len)
      dec_inputs.append(d_in)
      dec_targets.append(d_out)
      dec_lens.append(len_out)

  batch_dict = {"enc_inputs":   np.array(enc_inputs),
                "enc_lens":     np.array(enc_lens),
                "enc_targets":  np.array(enc_targets),
                "dec_bow":      np.array(dec_bow),
                "dec_bow_len":  np.array(dec_bow_len),
                "dec_inputs":   np.array(dec_inputs),
                "dec_targets":  np.array(dec_targets),
                "dec_lens":     np.array(dec_lens)}
  return batch_dict

def build_batch_seq2seq_eval(data_batch, len_batch):
  """Build an evaluation batch"""
  enc_inputs = []
  inp_lens = []
  references = []
  ref_lens = []
  
  for st, slen in zip(data_batch, len_batch):
    inp = st[0][: -1]
    len_inp = slen[0]
    ref = [s_[1: s_len_] for s_, s_len_ in zip(st[1:], slen[1:])]

    enc_inputs.append(inp)
    inp_lens.append(len_inp)
    references.append(ref)
    
  batch_dict = {"enc_inputs": np.array(enc_inputs),
                "inp_lens": np.array(inp_lens),
                "references": references}
                # "raw": data_batch,
                # "raw_len": len_batch}
  return batch_dict

def build_batch_seq2seq(data_batch, len_batch, use_mscoco14=False):
  """Build a batch of data for the sequence to sequence model"""
  enc_inputs = []
  dec_inputs = []
  targets = []
  inp_lens = []
  out_lens = []

  for st, slen in zip(data_batch, len_batch):
    num_para = len(st)
    if(use_mscoco14): iter_range = [0, 2]
    else: iter_range = range(num_para)
    for i in iter_range:
      j = (i + 1) % num_para
      inp = st[i][: -1]
      d_in = st[j][: -1]
      d_out = st[j][1: ]
      len_inp = slen[i]
      len_out = slen[j]

      enc_inputs.append(inp)
      dec_inputs.append(d_in)
      targets.append(d_out)
      inp_lens.append(len_inp)
      out_lens.append(len_out)

  # print('enc_inputs.shape:', np.array(enc_inputs).shape)
  batch_dict = {"enc_inputs": np.array(enc_inputs), 
                "dec_inputs": np.array(dec_inputs), 
                "targets": np.array(targets), 
                "inp_lens": np.array(inp_lens), 
                "out_lens": np.array(out_lens) }
  return batch_dict

def train_dev_split(dataset_name, train_sets, ratio=0.8):
  """Suffle the dataset and split the training set"""
  print("Splitting training and dev set ... ")

  if(dataset_name == "mscoco"): 
    train_index_file = "mscoco_train_index.txt"
    with open(train_index_file) as fd:
      train_index = set([int(l[:-1]) for l in fd.readlines()])

    train, dev = [], []
    for i in range(len(train_sets)):
      if(i in train_index): train.append(train_sets[i])
      else: dev.append(train_sets[i])

  if(dataset_name == "quora"): 
    train_index_file = "quora_train_index.txt"
    with open(train_index_file) as fd:
      train_index = set([int(l[:-1]) for l in fd.readlines()])

    dev_index_file = "quora_dev_index.txt"
    with open(dev_index_file) as fd:
      dev_index = set([int(l[:-1]) for l in fd.readlines()])

    train, dev = [], []
    for i in range(len(train_sets)):
      if(i in train_index): train.append(train_sets[i])
      elif(i in dev_index): dev.append(train_sets[i])

  elif(dataset_name == 'mscoco14'):
    with open('mscoco14_val_index.txt') as fd:
      val_index = set([int(l[:-1]) for l in fd])
    train, dev = [], []
    for i in range(len(train_sets)):
      if(i in val_index): dev.append(train_sets[i])
      else: train.append(train_sets[i])

  print("Size of training set: %d" % len(train))
  print("Size of test set: %d" % len(dev))
  return train, dev

class Dataset(object):
  """The dataset class, read the raw data, process into intermediate 
  representation, and load the intermediate as batcher"""

  def __init__(self, config):
    """Initialize the dataset configuration"""
    self.dataset = config.dataset
    self.dataset_path = config.dataset_path[self.dataset]
    self.max_sent_len = config.max_sent_len[self.dataset]
    self.max_enc_bow = config.max_enc_bow
    self.max_dec_bow = config.max_dec_bow
    self.bow_pred_method = config.bow_pred_method
    self.predict_source_bow = config.predict_source_bow
    self.single_ref = config.single_ref
    self.compare_outputs = config.compare_outputs

    self.stop_words = set(stopwords.words('english'))

    self.word2id = None
    self.id2word = None
    self.pad_id = -1 
    self.start_id = -1 
    self.eos_id = -1
    self.unk_id = -1

    self._dataset = {"train": None, "dev": None, "test": None}
    self._sent_lens = {"train": None, "dev": None, "test": None}
    self._ptr = {"train": 0, "dev": 0, "test": 0}
    return 

  @property
  def vocab_size(self): return len(self.word2id)

  def dataset_size(self, setname):
    return len(self._dataset[setname])
  
  def num_batches(self, batch_size, setname):
    setsize = self.dataset_size(setname)
    num_batches = setsize // batch_size + 1
    return num_batches

  def build(self):
    """Build the dataset to intermediate representation
    
    The data processing pipeline: 
    * read raw file 
    * calculate corpus statistics
    * split training and validation 
    * build vocabulary
    * normalize the text 
    """
    # read training sentences
    if(self.dataset == "mscoco"):
      train_sentences = mscoco_read_json(self.dataset_path["train"])
    elif(self.dataset == 'mscoco14'):
      train_sentences = mscoco_read_json(self.dataset_path["train"])
    elif(self.dataset == "quora"):
      train_sentences = quora_read(self.dataset_path["train"])

    # corpus_statistics(train_sentences)

    train_sentences, dev_sentences = train_dev_split(
      self.dataset, train_sentences) 

    word2id, id2word = get_vocab(train_sentences)

    train_sentences, train_lens = normalize(
      train_sentences, word2id, self.max_sent_len)
    dev_sentences, dev_lens = normalize(
      dev_sentences, word2id, self.max_sent_len)

    # test_sentences = mscoco_read_json(self.dataset_path["test"])
    # test_sentences, test_lens = normalize(
    #   test_sentences, word2id, self.max_sent_len)

    self.word2id = word2id
    self.id2word = id2word
    self.start_id = word2id["_GOO"]
    self.eos_id   = word2id["_EOS"]
    self.unk_id   = word2id["_UNK"]
    self.pad_id   = word2id["_PAD"]
    self.stop_words = set(
      [word2id[w] if(w in word2id) else self.pad_id for w in self.stop_words])
    self.stop_words |= set(
      [self.start_id, self.eos_id, self.unk_id, self.pad_id])

    self._dataset["train"] = train_sentences
    self._dataset["test"] = dev_sentences
    # self._dataset["test"] = test_sentences
    self._sent_lens["train"] = train_lens
    self._sent_lens["test"] = dev_lens
    # self._sent_lens["test"] = test_lens
    return 

  def next_batch(self, setname, batch_size, model):
    """Get next data batch
    
    Args:
      setname: a string, "train", "valid", or "test"
      batch_size: the size of the batch, an integer
      model_name: the name of the model, a string, different model use different
        data representations
    """
    bow_pred_method = self.bow_pred_method
    ptr = self._ptr[setname]
    data_batch = self._dataset[setname][ptr: ptr + batch_size]
    len_batch = self._sent_lens[setname][ptr: ptr + batch_size]

    if(setname == "train"):
      if(model in ["seq2seq", "lm"]):
        if(self.dataset == 'mscoco14'): mscoco14 = True
        else: mscoco14 = False
        batch_dict = build_batch_seq2seq(data_batch, len_batch, mscoco14)
      elif(model == "bow_seq2seq"):
        if(bow_pred_method == "seq2seq"):
          batch_dict = build_batch_seq2seq_bow2seq(data_batch, len_batch, 
            self.stop_words, self.max_enc_bow, self.max_dec_bow, self.pad_id)
        else:
          batch_dict = build_batch_bow_seq2seq(data_batch, len_batch, 
            self.stop_words, self.max_enc_bow, self.max_dec_bow, self.pad_id)
      elif(model == "latent_bow"):
        batch_dict = build_batch_bow_seq2seq(data_batch, len_batch, 
          self.stop_words, self.max_enc_bow, self.max_dec_bow, self.pad_id,
          self.predict_source_bow)
      elif(model == "vae"):
        batch_dict = build_batch_vae(data_batch, len_batch)
    else: # evaluation
      if(model == "seq2seq"):
        batch_dict_infer = build_batch_seq2seq_eval(data_batch, len_batch)
        # batch_dict_valid = build_batch_seq2seq(data_batch, len_batch)
        # batch_dict = (batch_dict_infer, batch_dict_valid)
        batch_dict = batch_dict_infer
      elif(model == "lm"):
        batch_dict = build_batch_seq2seq(data_batch, len_batch)
      elif(model == "bow_seq2seq"):
        if(bow_pred_method == "seq2seq"):
          batch_dict = build_batch_seq2seq_bow2seq(data_batch, len_batch, 
            self.stop_words, self.max_enc_bow, self.max_dec_bow, self.pad_id)
        else:
          batch_dict = build_batch_bow_seq2seq_eval(data_batch, len_batch,
            self.stop_words, self.max_enc_bow, self.max_dec_bow, self.pad_id)
      elif(model == "latent_bow"):
        batch_dict = build_batch_bow_seq2seq_eval(data_batch, len_batch,
          self.stop_words, self.max_enc_bow, self.max_dec_bow, self.pad_id,
          self.single_ref)

    ptr += batch_size
    if(ptr == self.dataset_size(setname)): 
      ptr = 0
    if(ptr + batch_size > self.dataset_size(setname)):
      ptr = self.dataset_size(setname) - batch_size
    self._ptr[setname] = ptr
    return batch_dict

  def decode_sent(self, sent, sent_len=-1, prob=None):
    """Decode the sentence index"""
    s_out = []
    is_break = False
    for wi, wid in enumerate(sent[:sent_len]):
      if(is_break): break
      w = self.id2word[wid]
      if(w == "_EOS"): 
        is_break = True
      s_out.append(w)
      if(prob is not None): s_out.append("(%.3f) " % prob[wi])
    return " ".join(s_out)

  def decode_neighbor(self, sent, neighbor_ind, neighbor_prob, sent_len=-1):
    """Decode the predicted sentence neighbot"""
    s_out = "" 
    is_break = False
    for wid, nb, np in zip(
      sent[: sent_len], neighbor_ind[: sent_len], neighbor_prob[: sent_len]):
      if(is_break): break
      w = self.id2word[wid]
      s_out += "%s: " % w
      for b, p in zip(nb, np):
        bw = self.id2word[b]
        s_out += "%s(%.4f), " % (bw, p)
      s_out += "\n"

      if(w == "_EOS"): 
        is_break = True
    return s_out

  def print_predict(self, 
    model_name, output_dict, batch_dict, fd=None, num_cases=6):
    """Print out the prediction"""
    if(model_name == "seq2seq"): 
      self.print_predict_seq2seq(output_dict, batch_dict, fd, num_cases)
    elif(model_name == "bow_seq2seq"):
      self.print_predict_bow_seq2seq(output_dict, batch_dict)
    elif(model_name == "latent_bow"):
      if(self.compare_outputs): 
        self.print_predict_seq2seq(output_dict, batch_dict, fd, num_cases)
      else: 
        self.print_predict_latent_bow(output_dict, batch_dict, fd)
    return

  def print_predict_latent_bow(self, output_dict, batch_dict, fd=None):
    """Print the prediction, latent bow model
    
    Things to print
    * The input sentence, 
    * The predicted bow and their probabilities
    * The sampled bow and their probabilities
    * The predicted sentences
    * The references
    """
    if(fd == None): print_range = 5
    else: print_range = len(output_dict["dec_predict"])
    for i in range(print_range):
      out = "inputs:\n"
      out += "    " + self.decode_sent(batch_dict["enc_inputs"][i]) + "\n"
      out += "input neighbors:\n"
      out += self.decode_neighbor(batch_dict["enc_inputs"][i], 
        output_dict["seq_neighbor_ind"][i], output_dict["seq_neighbor_prob"][i])
      out += "enc_output_bow:\n"
      out += "    " + self.decode_sent(output_dict["bow_pred_ind"][i]) + "\n"
      out += "enc_sampled_memory:\n"
      out += "    " + self.decode_sent(output_dict["sample_memory_ind"][i]) + "\n"
      out += "dec_outputs:\n"
      out += "    " + self.decode_sent(output_dict["dec_predict"][i]) + "\n"
      out += "references:\n"
      for r in batch_dict["references"][i]:
        out += "    " + self.decode_sent(r) + "\n"
      if(fd != None): fd.write(out + "\n")
      else: print(out)
    return

  def print_gumbel(self, output_dict, batch_dict, fd=None):
    """Print the gumbel sampes """
    if(fd == None): print_range = 5
    else: print_range = len(output_dict[0]["dec_predict"])
    for i in range(print_range):
      dec_out_i = []
      for j in  range(len(output_dict)):
        dec_out_i.append(output_dict[j]["dec_predict"][i])
      dec_out_i = np.array(dec_out_i)
      if(np.all((dec_out_i - dec_out_i[0]) == 0)): continue
      
      out = "inputs:\n"
      out += "    " + self.decode_sent(batch_dict["enc_inputs"][i]) + "\n\n"
      for j in range(len(output_dict)):
        # out += "sample %d, input neighbors:\n" % j
        # out += self.decode_neighbor(batch_dict["enc_inputs"][i], 
        #                             output_dict[j]["seq_neighbor_ind"][i], 
        #                             output_dict[j]["seq_neighbor_prob"][i])
        # out += "sample %d, enc_output_bow:\n" % j
        # out += "    " + self.decode_sent(output_dict[j]["bow_pred_ind"][i]) + "\n"
        out += "sample %d, enc_sampled_memory:\n" % j
        out += "    " + self.decode_sent(
          output_dict[j]["sample_memory_ind"][i], 
          prob=output_dict[j]["sample_memory_prob"][i]) + "\n"
        out += "sample %d, dec_outputs:\n" % j
        out += "    " + self.decode_sent(
          output_dict[j]["dec_predict"][i]) + "\n\n"
      # out += "references:\n"
      # for r in batch_dict["references"][i]:
      #   out += "    " + self.decode_sent(r) + "\n"
      out += "----\n"
      if(fd is not None): fd.write(out + "\n")
      else: print(out)
    return

  def print_predict_bow_seq2seq(self, output_dict, batch_dict):
    """Print the predicted sentences for the bag of words - sequence to 
    sequence model

    Things to print:
      * The input sentence 
      * The predicted bow 
      * The sample from the bow 
      * The predicted sentence 
      * The references
    """
    enc_sentences = batch_dict["enc_inputs"]
    enc_outputs = output_dict["pred_ind"]
    dec_samples = output_dict["dec_sample_bow"]
    dec_outputs = output_dict["dec_predict"]
    references = batch_dict["references"]

    for i, (es, eo, ds, do, rf) in enumerate(zip(enc_sentences, enc_outputs, 
      dec_samples, dec_outputs, references)):
      print("inputs:")
      print("    " + self.decode_sent(es))
      print("enc_outputs:")
      print("    " + self.decode_sent(eo))
      print("dec_samples:")
      print("    " + self.decode_sent(ds))
      print("dec_outputs:")
      print("    " + self.decode_sent(do))
      print("reference:")
      for r in rf:
        print("    " + self.decode_sent(r))
      print("")

      if(i == 5): break
    return

  def print_predict_seq2seq(
    self, output_dict, batch_dict, fd=None, num_cases=6):
    """Print the predicted sentences for the sequence to sequence model"""
    predict = output_dict["dec_predict"]
    inputs = batch_dict["enc_inputs"]
    references = batch_dict["references"]
    batch_size = output_dict["dec_predict"].shape[0]
    for i in range(batch_size):
      str_out = 'inputs:\n'
      str_out += self.decode_sent(inputs[i]) + '\n'
      str_out += 'outputs:\n'
      str_out += self.decode_sent(predict[i]) + '\n'
      str_out += "references:\n"
      for r in references[i]:
        str_out += self.decode_sent(r) + '\n'
      str_out += '----\n'
      if(i < num_cases): print(str_out)
      if(fd is not None): fd.write(str_out)
    return 
  
  def print_predict_seq2paraphrase(self, output_dict, batch_dict, num_cases=3):
    """Print the predicted sentences, sequence to k sequence model (given a 
      sentence, predict all k possible paraphrases)"""
    inputs = batch_dict["enc_inputs"][:3]
    references = batch_dict["references"][:3]
    for i in range(num_cases):
      print("inputs:")
      print("    " + self.decode_sent(inputs[i]))
      pred_para = output_dict["enc_infer_pred"][i]
      print("paraphrase outputs:")
      for p in pred_para:
        print("    " + self.decode_sent(p))
      print("references:")
      for r in references[i]:
        print("    " + self.decode_sent(r))
      print("")
    return 

  def print_random_walk(self, random_walk_outputs, batch_dict, num_cases=3):
    """Print the random walk outputs"""
    inputs = batch_dict["enc_inputs"][:3]
    references = batch_dict["references"][:3]
    for i in range(num_cases):
      print("inputs:")
      print("    " + self.decode_sent(inputs[i]))
      for d in random_walk_outputs:
        print("->")
        print("    " + self.decode_sent(d["predict"][i]))
      print("references:")
      for r in references[i]:
        print("    " + self.decode_sent(r))
      print("")
    return

  def print_bow(self, output_dict, batch_dict):
    """Print the bow prediction: the input sentence, the target bow, and the 
    predicted bow. """
    enc_sentences = batch_dict["enc_inputs"]
    enc_targets = batch_dict["enc_targets"]
    enc_outputs = output_dict["pred_ind"]

    def _decode_set(s, shared):
      output = []
      for si in s:
        if(si in shared): output.append("[" + self.id2word[si] + "]")
        else: output.append(self.id2word[si])
      return

    for i, (s, t, o) in enumerate(zip(enc_sentences, enc_targets, enc_outputs)):
      if(i in [0, 1, 5, 6, 10, 11, 15, 16]):
        print("inputs:")
        print("    " + self.decode_sent(s))
        shared = set(t) & set(o)
        print("targets:")
        print("    " + _decode_set(set(t) - set([self.pad_id]), shared))
        print("outputs:")
        print("    " + _decode_set(set(o), shared))
        print("")
      if(i == 16): break
    return

"""The NLP data cleaning pipeline

Yao Fu, Columbia University 
yao.fu@columabia.edu
THU MAY 09TH 2019 
"""
import numpy as np 

import nltk 
from nltk.corpus import stopwords
from collections import Counter

def normalize(sentence_sets, word2id, max_sent_len):
  """Normalize the sentences by the following procedure
  - word to index 
  - add unk
  - pad/ cut the sentence length
  - record the sentence length

  Args: 
    sentence_sets: the set of sentence paraphrase, a list of sentence list
    word2id: word index, a dictionary
    max_sent_len: maximum sentence length, a integer
  """
  sent_sets = []
  sent_len_sets = []
  max_sent_len = max_sent_len + 1

  for st in sentence_sets:
    st_ = []
    st_len = []
    for s in st:
      s_ = [word2id["_GOO"]]
      for w in s:
        if(w in word2id): s_.append(word2id[w])
        else: s_.append(word2id["_UNK"])
      s_.append(word2id["_EOS"])
      
      s_ = s_[: max_sent_len]
      if(len(s_) < max_sent_len):
        s_len = len(s_) - 1
        for i in range(max_sent_len - len(s_)): s_.append(word2id["_PAD"])
      else: 
        s_[-1] = word2id["_EOS"]
        s_len = max_sent_len - 1

      st_.append(s_)
      st_len.append(s_len)
    
    sent_sets.append(st_)
    sent_len_sets.append(st_len)
  return sent_sets, sent_len_sets

def corpus_statistics(sentence_sets, vocab_size_threshold=5):
  """Calculate basic corpus statistics"""
  print("Calculating basic corpus statistics .. ")

  stop_words = set(stopwords.words('english'))

  # size of paraphrase sets
  paraphrase_size = [len(st) for st in sentence_sets]
  paraphrase_size = Counter(paraphrase_size)
  print("paraphrase size, %d different types:" % (len(paraphrase_size)))
  print(paraphrase_size.most_common(10))

  # sentence length
  sentence_lens = []
  sentence_bow_len = [] 
  paraphrase_bow_len = []
  for st in sentence_sets:
    sentence_lens.extend([len(s) for s in st])
    st_bow = set()
    for s in st:
      s_ = set(s) - stop_words
      sentence_bow_len.append(len(s_))
      st_bow |= s_
    paraphrase_bow_len.append(len(st_bow))

  sent_len_percentile = np.percentile(sentence_lens, [80, 90, 95])
  print("sentence length percentile:")
  print(sent_len_percentile)

  sentence_bow_percentile = np.percentile(sentence_bow_len, [80, 90, 95])
  print("sentence bow length percentile")
  print(sentence_bow_percentile)

  paraphrase_bow_percentile = np.percentile(paraphrase_bow_len, [80, 90, 95])
  print("paraphrase bow length percentile")
  print(paraphrase_bow_percentile)

  # vocabulary
  vocab = []
  for st in sentence_sets:
    for s in st:
      vocab.extend(s)
  vocab = Counter(vocab)
  print("vocabulary size: %d" % len(vocab))
  vocab_truncate = [w for w in vocab if vocab[w] >= vocab_size_threshold]
  print("vocabulary size, occurance >= 5: %d" % len(vocab_truncate))
  return 

def get_vocab(training_set, vocab_size_threshold=5):
  """Get the vocabulary from the training set"""
  vocab = []
  for st in training_set:
    for s in st:
      vocab.extend(s)

  vocab = Counter(vocab)
  vocab_truncate = [w for w in vocab if vocab[w] >= vocab_size_threshold]
  
  word2id = {"_GOO": 0, "_EOS": 1, "_PAD": 2, "_UNK": 3}
  id2word = {0: "_GOO", 1: "_EOS", 2: "_PAD", 3: "_UNK"}

  i = len(word2id)
  for w in vocab_truncate:
    word2id[w] = i
    id2word[i] = w
    i += 1
  
  assert(len(word2id) == len(id2word))
  print("vocabulary size: %d" % len(word2id))
  return word2id, id2word
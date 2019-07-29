"""The sequence to bow to sequence model, hard version 

Yao Fu, Columbia University 
yao.fu@columbia.edu
TUE APR 23RD 2019 
"""

import numpy as np 
import tensorflow as tf 

from tensorflow.nn.rnn_cell import LSTMStateTuple
from tensorflow.contrib import slim
from seq2seq import create_cell
from decoder import decoding_infer, decoding_train, attention, decode

## Model functions
def _enc_target_list_to_khot(enc_targets, vocab_size, pad_id):
  """Convert a batch of target list to k-hot vectors"""
  enc_targets = tf.one_hot(enc_targets, vocab_size) # [B, BOW, V]
  enc_targets = tf.reduce_sum(enc_targets, axis=1) # [B, V]
  enc_target_mask = 1. - tf.one_hot(
    [pad_id], vocab_size, dtype=tf.float32)
  enc_targets *= enc_target_mask
  return enc_targets

def bow_train_monitor(
  bow_topk_prob, pred_ind, vocab_size, batch_size, enc_targets):
  """Precision and recall for the bow model, as well as their supports"""
  pred_one_hot = tf.one_hot(pred_ind, vocab_size)
  pred_one_hot = tf.reduce_sum(pred_one_hot, axis=1)
  pred_one_hot = tf.cast(pred_one_hot, tf.bool)
  pred_topk_support = tf.reduce_sum(tf.cast(pred_one_hot, tf.float32))

  pred_confident = tf.cast(bow_topk_prob > 0.5, tf.bool) # approximate 
  pred_confident_support = tf.reduce_sum(tf.cast(pred_confident, tf.float32)) 
  predict_average_confident = pred_confident_support / \
    tf.cast(batch_size, tf.float32)

  enc_targets_ = tf.cast(enc_targets, tf.bool)
  pred_overlap_topk = tf.reduce_sum(
    tf.cast(tf.logical_and(pred_one_hot, enc_targets_), tf.float32))

  pred_overlap_confident = tf.reduce_sum(
    tf.cast(tf.logical_and(pred_confident, enc_targets_), tf.float32))

  target_support = tf.reduce_sum(enc_targets)
  precision_confident = pred_overlap_confident / (pred_confident_support + 1)
  recall_confident = pred_overlap_confident / (target_support + 1)
  precision_topk = pred_overlap_topk / (pred_topk_support + 1)
  recall_topk = pred_overlap_topk / (target_support + 1)
  target_average = target_support / tf.cast(batch_size, tf.float32)

  metric_dict = { "pred_overlap_topk": pred_overlap_topk,
                  "pred_overlap_confident": pred_overlap_confident,

                  "pred_topk_support": pred_topk_support, 
                  "pred_confident_support": pred_confident_support,
                  "target_support": target_support,

                  "predict_average_confident": predict_average_confident,
                  "target_average": target_average,

                  # "pred_prob": pred_prob,
                  # "pred_prob_unnorm": pred_prob_unnorm, 
                  # "pred_ind": pred_ind, 

                  "precision_confident": precision_confident, 
                  "recall_confident": recall_confident, 
                  "precision_topk": precision_topk, 
                  "recall_topk": recall_topk}
  return metric_dict

def nll_loss(target_ind, pred):
  """nll loss"""
  batch_size = tf.shape(pred)[0]
  target_size = tf.shape(target_ind)[1]
  ind_ = tf.tile(tf.range(batch_size), [target_size])
  ind_ = tf.reshape(ind_, [target_size, batch_size])
  ind_ = tf.transpose(ind_, [1, 0])
  ind_ = tf.expand_dims(ind_, [2])
  target_ind_ = tf.expand_dims(target_ind, [2])
  ind_ = tf.concat([ind_, target_ind_], 2)
  output = tf.gather_nd(pred, ind_)

  loss = tf.reduce_mean(-tf.log(output + 1e-6))
  return loss

def enc_loss_fn(bow_loss_fn, enc_targets, bow_topk_prob, max_enc_bow):
  """Different encoder loss wrapper"""
  # NLL loss 
  if(bow_loss_fn == "nll"):
    enc_loss = - enc_targets * tf.log(bow_topk_prob + 1e-6)
    enc_loss_norm = tf.reduce_sum(enc_targets, 1) + 1.0
    enc_loss = tf.reduce_mean(tf.reduce_sum(enc_loss, 1) / enc_loss_norm)

  # cross entropy loss 
  # normalize -- This is not so strict, but be it for now 
  elif(bow_loss_fn == "crossent"):
    bow_topk_prob /= float(max_enc_bow)
    enc_loss = - (enc_targets * tf.log(bow_topk_prob + 1e-6) + 
      (1 - enc_targets) * tf.log(1 - bow_topk_prob + 1e-6))
    enc_loss = tf.reduce_mean(
      tf.reduce_sum(enc_loss, axis=1) / tf.reduce_sum(enc_targets, axis=1))

  # L1 distance loss 
  # L1 distance = total variation of the two distributions 
  elif(bow_loss_fn == "l1"):
    enc_loss = tf.losses.absolute_difference(enc_targets, bow_topk_prob)
  return enc_loss

def bow_predict_seq_tag(vocab_size,
                        enc_batch_size,
                        enc_outputs, 
                        enc_lens, 
                        max_len, 
                        is_gumbel=False,
                        tau=0.5,
                        max_src2tgt_word=3):
  """bow prediction as sequence tagging
  
  Let each word from the source sentence predict its k nearest neighbors 
  """
  bow_topk_prob = tf.zeros([enc_batch_size, vocab_size])
  gumbel_topk_prob = tf.zeros([enc_batch_size, vocab_size])
  seq_neighbor_ind = []
  seq_neighbor_prob = []

  def _sample_gumbel(shape, eps=1e-20):
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)

  for i in range(max_src2tgt_word):
    bow_trans = tf.layers.Dense(500, name="bow_src2tgt_trans_%d" % i,
      kernel_initializer=tf.random_normal_initializer(stddev=0.05),
      bias_initializer=tf.constant_initializer(0.))
    bow_proj = tf.layers.Dense(vocab_size, name="bow_src2tgt_proj_%d" % i,
      kernel_initializer=tf.random_normal_initializer(stddev=0.05),
      bias_initializer=tf.constant_initializer(0.))
    bow_logits = bow_proj(bow_trans(enc_outputs)) # [B, T, S] -> [B, T, V]
      
    # mixture of softmax probability 
    bow_prob = tf.nn.softmax(bow_logits)
    pred_mask = tf.expand_dims(
      tf.sequence_mask(enc_lens, max_len, tf.float32), [2]) # [B, T, 1]
    bow_prob *= pred_mask # [B, T, V]
    bow_topk_prob += tf.reduce_sum(bow_prob, 1)

    # record neighbor prediction
    neighbor_ind = tf.argmax(bow_prob, 2) # [B, T]
    seq_neighbor_ind.append(neighbor_ind)
    neighbor_prob = tf.reduce_max(bow_prob, 2) # [B, T]
    seq_neighbor_prob.append(neighbor_prob) 

    # gumbel reparameterization
    if(is_gumbel): 
      print("Using gumbel reparametrization ... ")
      gumbel_prob = tf.nn.softmax(
        (bow_logits + _sample_gumbel(tf.shape(bow_logits))) / tau)
      gumbel_prob *= pred_mask
      gumbel_topk_prob += tf.reduce_sum(gumbel_prob, 1)
    else: 
      print("Not using gumbel reparametrization ... ")
      gumbel_topk_prob += tf.reduce_sum(bow_prob, 1)

  seq_neighbor_ind = tf.stack(seq_neighbor_ind, 2) # [B, T, N]
  seq_neighbor_prob = tf.stack(seq_neighbor_prob, 2) # [B, T, N]
  return bow_topk_prob, gumbel_topk_prob, seq_neighbor_ind, seq_neighbor_prob


def bow_seq2seq_metrics(enc_targets, enc_seq_pred, vocab_size, pad_id):
  """Precision and recall"""
  enc_seq_pred = tf.one_hot(enc_seq_pred, vocab_size) # [B, P, T, V] 
  pad_mask = 1. - tf.one_hot([pad_id], vocab_size, dtype=tf.float32) # [1, V]
  enc_seq_pred = pad_mask * tf.reduce_sum(enc_seq_pred, axis=[1, 2]) 
  enc_seq_pred = tf.cast(enc_seq_pred, tf.bool)

  enc_targets_ = tf.cast(enc_targets, tf.bool)
  overlap = tf.cast(tf.logical_and(enc_seq_pred, enc_targets_), tf.float32)
  overlap = tf.reduce_sum(overlap)

  pred_support = tf.reduce_sum(tf.cast(enc_seq_pred, tf.float32))
  target_support = tf.reduce_sum(enc_targets)

  prec = overlap / pred_support
  recl = overlap / target_support
  return overlap, pred_support, target_support, prec, recl

def bow_predict_seq2seq(enc_seq2seq_inputs, 
                        enc_seq2seq_targets,
                        enc_seq2seq_lens, 
                        embedding_matrix, 
                        enc_outputs, 
                        enc_state, 
                        enc_layers,
                        num_paraphrase,  
                        max_len, 
                        enc_lens, 
                        batch_size,
                        vocab_size, 
                        state_size, 
                        drop_out, 
                        dec_start_id):
  """bow prediction as sequence to sequence"""

  enc_seq2seq_inputs = tf.nn.embedding_lookup(
    embedding_matrix, enc_seq2seq_inputs) 
  # [B, P, T, S] -> [P, B, T, S]
  enc_seq2seq_inputs = tf.transpose(enc_seq2seq_inputs, [1, 0, 2, 3]) 
  # [B, P, T] -> [P, B, T]
  enc_seq2seq_targets = tf.transpose(enc_seq2seq_targets, [1, 0, 2]) 
  # [B, P] -> [P, B]
  enc_seq2seq_lens = tf.transpose(enc_seq2seq_lens, [1, 0])
  
  init_state = enc_state
  enc_pred_loss = 0.0
  bow_topk_prob = tf.zeros([batch_size, vocab_size])
  enc_infer_pred = []
  for i in range(num_paraphrase):

    # encoder prediction cell 
    enc_pred_cell = [create_cell("enc_pred_p_%d_l_%d" % (i, j), state_size, drop_out) 
          for j in range(enc_layers)]
    enc_pred_cell = tf.nn.rnn_cell.MultiRNNCell(enc_pred_cell)

    # projection 
    enc_pred_proj = tf.layers.Dense(vocab_size, name="enc_pred_proj",
        kernel_initializer=tf.random_normal_initializer(stddev=0.05),
        bias_initializer=tf.constant_initializer(0.))

    # greedy decoding and training 
    _, enc_seq_predict = decoding_infer(dec_start_id,
                                        enc_pred_cell,
                                        enc_pred_proj,
                                        embedding_matrix,
                                        init_state,
                                        enc_outputs,
                                        batch_size,
                                        max_len,
                                        enc_lens,
                                        max_len,
                                        is_attn=True)
    enc_infer_pred.append(enc_seq_predict)

    enc_pred_inputs = enc_seq2seq_inputs[i]
    enc_seq_train = decoding_train( enc_pred_inputs, 
                                    enc_pred_cell, 
                                    init_state, 
                                    enc_outputs,  
                                    max_len, 
                                    enc_lens, 
                                    max_len,
                                    is_attn=True)
    enc_seq_train_logits = enc_pred_proj(enc_seq_train)

    # sequence to sequence loss 
    enc_seq_mask = tf.sequence_mask(
      enc_seq2seq_lens[i], max_len, dtype=tf.float32)
    enc_seq_loss = tf.contrib.seq2seq.sequence_loss(
      enc_seq_train_logits, enc_seq2seq_targets[i], enc_seq_mask)
    enc_pred_loss += enc_seq_loss

    # prediction probability 
    enc_pred_prob = tf.nn.softmax(enc_seq_train_logits) # [B, T, V]
    enc_pred_prob *= tf.expand_dims(enc_seq_mask, [2]) # [B, T, 1]
    enc_pred_prob = tf.reduce_sum(enc_pred_prob, axis=1) # [B, V]
    # NOTE: prob of certain words will be repeatedly calculated
    bow_topk_prob += enc_pred_prob 

  enc_pred_loss /= num_paraphrase

  enc_infer_pred = tf.stack(enc_infer_pred) # [P, B, T]
  enc_infer_pred = tf.transpose(enc_infer_pred, [1, 0, 2]) # [B, P, T]
  return bow_topk_prob, enc_pred_loss, enc_infer_pred

def bow_predict_mix_softmax(enc_batch_size, vocab_size, max_enc_bow, enc_state):
  """bow prediction with mixture of softmax"""
  bow_topk_prob = tf.zeros([enc_batch_size, vocab_size])

  # The mixture of softmax approach 
  for i in range(max_enc_bow):
    bow_proj = tf.layers.Dense(vocab_size, name="bow_proj_%d" % i,
      kernel_initializer=tf.random_normal_initializer(stddev=0.05),
      bias_initializer=tf.constant_initializer(0.))
    bow_logits = bow_proj(enc_state[1].h)
    bow_prob_i = tf.nn.softmax(bow_logits, axis=1)
    bow_topk_prob += bow_prob_i
  return bow_topk_prob

## Model class 

class BowSeq2seq(object):
  """The ngram sequence to sequence model

  Put the encoder and the decoder into the same class, during training, train 
  them seperatele, during prediction, put the sampling procedure in tensorflow 
  for better speed
  """

  def __init__(self, config):
    """Initialization, just copy the configuration"""
    self.mode = config.model_mode
    self.vocab_size = config.vocab_size
    self.max_enc_bow = config.max_enc_bow
    self.max_dec_bow = config.max_dec_bow
    self.sample_size = config.sample_size
    self.source_sample_ratio = config.source_sample_ratio
    self.bow_pred_method = config.bow_pred_method
    self.state_size = config.state_size
    self.enc_layers = config.enc_layers
    self.learning_rate = config.learning_rate
    self.learning_rate_enc = config.learning_rate_enc
    self.learning_rate_dec = config.learning_rate_dec
    self.optimizer = config.optimizer
    self.dec_start_id = config.dec_start_id
    self.dec_end_id = config.dec_end_id
    self.pad_id = config.pad_id
    self.is_attn = config.is_attn
    self.stop_words = config.stop_words
    self.bow_loss_fn = config.bow_loss_fn
    self.num_paraphrase = config.num_paraphrase
    self.sampling_method = config.sampling_method
    self.topk_sampling_size = config.topk_sampling_size
    self.model_name = config.model_name
    self.is_cheat = config.is_cheat
    return 

  def build(self):
    """Build the model """
    print("Building the bow - sequence to sequence model ... ")

    vocab_size = self.vocab_size
    state_size = self.state_size
    enc_layers = self.enc_layers
    max_enc_bow = self.max_enc_bow
    num_paraphrase = self.num_paraphrase

    # Placeholders
    with tf.name_scope("placeholders"):
      enc_inputs = tf.placeholder(tf.int32, [None, None], "enc_inputs")
      enc_lens = tf.placeholder(tf.int32, [None], "enc_lens")
      self.drop_out = tf.placeholder(tf.float32, (), "drop_out")
      self.max_len = tf.placeholder(tf.int32, (), "max_len")
      dec_bow = tf.placeholder(tf.int32, [None, None], "dec_bow")
      dec_bow_len = tf.placeholder(tf.int32, [None], "dec_bow_len")

      self.enc_inputs = enc_inputs
      self.enc_lens = enc_lens
      self.dec_bow = dec_bow 
      self.dec_bow_len = dec_bow_len

      if(self.mode == "train"):
        enc_targets = tf.placeholder(tf.int32, [None, None], "enc_targets")
        enc_seq2seq_inputs = tf.placeholder(
          tf.int32, [None, num_paraphrase, None], "enc_seq2seq_inputs")
        enc_seq2seq_targets = tf.placeholder(
          tf.int32, [None, num_paraphrase, None], "enc_seq2seq_targets")
        enc_seq2seq_lens = tf.placeholder(
          tf.int32, [None, num_paraphrase], "enc_seq2seq_lens")

        dec_inputs = tf.placeholder(tf.int32, [None, None], "dec_inputs")
        dec_targets = tf.placeholder(tf.int32, [None, None], "dec_targets")
        dec_lens = tf.placeholder(tf.int32, [None], "dec_lens")

        self.enc_targets = enc_targets
        self.enc_seq2seq_inputs = enc_seq2seq_inputs
        self.enc_seq2seq_targets = enc_seq2seq_targets
        self.enc_seq2seq_lens = enc_seq2seq_lens
        self.dec_inputs = dec_inputs
        self.dec_targets = dec_targets
        self.dec_lens = dec_lens

    enc_batch_size = tf.shape(enc_inputs)[0]
    max_len = self.max_len

    dec_batch_size = tf.shape(dec_bow)[0]
    max_dec_bow = tf.shape(dec_bow)[1]

    # Embedding 
    with tf.variable_scope("embeddings"):
      embedding_matrix = tf.get_variable(
        name="embedding_matrix", 
        shape=[vocab_size, state_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=0.05))
      enc_inputs = tf.nn.embedding_lookup(embedding_matrix, enc_inputs)

      if(self.mode == "train"): 
        dec_inputs = tf.nn.embedding_lookup(embedding_matrix, dec_inputs)
        dec_bow = tf.nn.embedding_lookup(embedding_matrix, dec_bow)

    # Encoder
    with tf.variable_scope("encoder"):
      # TODO: residual LSTM, layer normalization
      enc_cell = [create_cell("enc-%d" % i, state_size, self.drop_out) 
        for i in range(enc_layers)]
      enc_cell = tf.nn.rnn_cell.MultiRNNCell(enc_cell)
      enc_outputs, enc_state = tf.nn.dynamic_rnn(enc_cell, enc_inputs,
        sequence_length=enc_lens, dtype=tf.float32)

    # Encoder bow prediction
    with tf.variable_scope("bow_output"):
      if(self.bow_pred_method == "mix_softmax"): 
        bow_topk_prob = bow_predict_mix_softmax(
          enc_batch_size, vocab_size, max_enc_bow, enc_state)

      elif(self.bow_pred_method == "seq_tag"):
        bow_topk_prob, _, _, _ = bow_predict_seq_tag(
          vocab_size, enc_batch_size, enc_outputs, enc_lens, max_len)

      elif(self.bow_pred_method == "seq2seq"):
        bow_topk_prob, enc_seq2seq_loss, enc_infer_pred = \
                                    bow_predict_seq2seq(enc_seq2seq_inputs, 
                                                        enc_seq2seq_targets,
                                                        enc_seq2seq_lens, 
                                                        embedding_matrix,
                                                        enc_outputs,
                                                        enc_state,
                                                        enc_layers,
                                                        num_paraphrase,
                                                        max_len,
                                                        enc_lens,
                                                        enc_batch_size,
                                                        vocab_size,
                                                        state_size, 
                                                        self.drop_out, 
                                                        self.dec_start_id)
      
    with tf.variable_scope("enc_optimizer"):
      enc_optimizer = tf.train.AdamOptimizer(self.learning_rate_enc)

    with tf.name_scope("enc_output"):
      # top k prediction 
      pred_prob, pred_ind = tf.nn.top_k(bow_topk_prob, max_enc_bow)
      pred_prob_unnorm = pred_prob
      pred_prob /= tf.expand_dims(tf.reduce_sum(pred_prob, axis=1), [1])

      pred_prob_dec, pred_ind_dec = tf.nn.top_k(bow_topk_prob, self.sample_size)
      pred_prob_dec /= tf.expand_dims(tf.reduce_sum(pred_prob_dec, axis=1), [1])

      if(self.mode == "train"):
        with tf.name_scope("enc_loss"):
          # loss function 
          enc_targets = _enc_target_list_to_khot(
            enc_targets, vocab_size, self.pad_id)
          enc_bow_loss = enc_loss_fn(
            self.bow_loss_fn, enc_targets, bow_topk_prob, max_enc_bow)
          if(self.bow_pred_method == "seq2seq"): 
            # pure sequence to sequence for now 
            enc_loss = enc_seq2seq_loss + 0.0 * enc_bow_loss
          else: 
            enc_loss = enc_bow_loss
          enc_train_op = enc_optimizer.minimize(enc_loss)

        # prediction preformance monitor during training 
        # write this in a function 
        # TODO: top 10 recall 
        with tf.name_scope("train_output"):
          # encoder training output
          self.enc_train_output = { "enc_train_op": enc_train_op, 
                                    "enc_bow_loss": enc_bow_loss,
                                    "enc_loss": enc_loss}
          bow_metrics_dict = bow_train_monitor(
            bow_topk_prob, pred_ind, vocab_size, enc_batch_size, enc_targets)
          self.enc_train_output.update(bow_metrics_dict)

          if(self.bow_pred_method == "seq2seq"): 
            self.enc_train_output["enc_seq2seq_loss"] = enc_seq2seq_loss

      # encoder inference output
      with tf.name_scope("infer_output"):
        if(self.bow_pred_method == "seq2seq"): 
          (infer_overlap, infer_pred_support, infer_target_support, infer_prec, 
            infer_recl) = bow_seq2seq_metrics(
              enc_targets, enc_infer_pred, vocab_size, self.pad_id)
          self.enc_infer_output = { 
            "enc_infer_overlap": infer_overlap,
            "enc_infer_pred_support": infer_pred_support,
            "enc_infer_target_support": infer_target_support,
            "enc_infer_precision": infer_prec,
            "enc_infer_recall": infer_recl,
            "enc_infer_pred": enc_infer_pred}
        else:
          self.enc_infer_output = { "pred_prob": pred_prob,
                                    "pred_ind": pred_ind,
                                    "pred_prob_dec": pred_prob_dec,
                                    "pred_ind_dec": pred_ind_dec}
        

    # Decoder bow encoding
    # TODO: sampling from encoder topk prediction
    with tf.variable_scope("dec_bow_encoding"):
      dec_bow_mask = tf.expand_dims(
        tf.sequence_mask(dec_bow_len, max_dec_bow, dtype=tf.float32), [2])
      
      # TODO: transformer based encoding, but our primary goal is to test the 
      # effectiveness of sampling, so we skip it for now 
      dec_bow_enc = tf.reduce_mean(dec_bow_mask * dec_bow, axis = 1) # [B, S]

    with tf.variable_scope("decoder"):
      dec_cell = [create_cell("dec-%d" % i, state_size, self.drop_out) 
        for i in range(enc_layers)]
      dec_cell = tf.nn.rnn_cell.MultiRNNCell(dec_cell)

      dec_init_state = (LSTMStateTuple(dec_bow_enc, dec_bow_enc), 
                        LSTMStateTuple(dec_bow_enc, dec_bow_enc))
      dec_proj = tf.layers.Dense(vocab_size, name="dec_proj",
        kernel_initializer=tf.random_normal_initializer(stddev=0.05),
        bias_initializer=tf.constant_initializer(0.))
      dec_memory = dec_bow
      dec_mem_len = dec_bow_len
      dec_max_mem_len = max_dec_bow


      # greedy decoding
      # _, dec_outputs_predict = decoding_infer(self.dec_start_id,
      #                                         dec_cell,
      #                                         dec_proj,
      #                                         embedding_matrix,
      #                                         dec_init_state,
      #                                         dec_bow,
      #                                         dec_batch_size,
      #                                         max_len,
      #                                         dec_bow_len,
      #                                         max_dec_bow,
      #                                         self.is_attn)

      # if(self.mode == "train"):
      #   # training decoding
      #   dec_outputs_train = decoding_train( dec_inputs, 
      #                                       dec_cell, 
      #                                       dec_init_state, 
      #                                       dec_bow,  
      #                                       max_len, 
      #                                       dec_bow_len, 
      #                                       max_dec_bow,
      #                                       self.is_attn)
      #   dec_logits_train = dec_proj(dec_outputs_train)

      dec_outputs_predict, dec_logits_train = decode( 
        self.dec_start_id, dec_inputs, 
        dec_cell, dec_proj, embedding_matrix, 
        dec_init_state, dec_memory, dec_mem_len, dec_max_mem_len, 
        dec_batch_size, max_len, self.sampling_method, self.topk_sampling_size,
        state_size, multi_source=False)

    all_variables = slim.get_variables_to_restore()
    model_variables = [var for var in all_variables 
      if var.name.split("/")[0] == self.model_name]
    print("%s model, variable list:" % self.model_name)
    for v in model_variables: print("  %s" % v.name)
    self.model_saver = tf.train.Saver(model_variables, max_to_keep=3)  

    with tf.variable_scope("dec_optimizer"):
      dec_optimizer = tf.train.AdamOptimizer(self.learning_rate_dec)

    with tf.name_scope("dec_output"):
      if(self.mode == "train"):
        dec_mask = tf.sequence_mask(dec_lens, max_len, dtype=tf.float32)
        dec_loss = tf.contrib.seq2seq.sequence_loss(
          dec_logits_train, dec_targets, dec_mask)
        dec_train_op = dec_optimizer.minimize(dec_loss)

        self.dec_train_output = { "dec_train_op": dec_train_op, 
                                  "dec_loss": dec_loss}
    
      self.dec_infer_output = {"dec_predict": dec_outputs_predict}
    return 

  def train_step(self, sess, batch_dict, ei):
    """Single step training"""
    if(self.is_cheat == False):
      output_dict = self.enc_train_step(sess, batch_dict)

      if(self.bow_pred_method != "seq2seq"):
        dec_output = self.dec_train_step(sess, batch_dict)
        output_dict.update(dec_output)
        output_dict["loss"] = output_dict["enc_loss"] + output_dict["dec_loss"]
    else:
      output_dict = self.dec_train_step(sess, batch_dict)
    return output_dict

  def enc_train_step(self, sess, batch_dict):
    """Single step training on encoder"""
    max_len = batch_dict["enc_inputs"].shape[1]
    if(self.bow_pred_method == "seq2seq"):
      feed_dict = { self.enc_inputs: batch_dict["enc_inputs"],
                    self.enc_lens: batch_dict["enc_lens"],
                    self.enc_targets: batch_dict["enc_targets"], 
                    self.enc_seq2seq_inputs: batch_dict["enc_seq2seq_inputs"],
                    self.enc_seq2seq_targets: batch_dict["enc_seq2seq_targets"],
                    self.enc_seq2seq_lens: batch_dict["enc_seq2seq_lens"],
                    self.dec_bow: batch_dict["dec_bow"],
                    self.max_len: max_len,
                    self.drop_out: 0.3}
    else:
      feed_dict = { self.enc_inputs: batch_dict["enc_inputs"],
                    self.enc_lens: batch_dict["enc_lens"],
                    self.enc_targets: batch_dict["enc_targets"], 
                    self.dec_bow: batch_dict["dec_bow"],
                    self.max_len: max_len,
                    self.drop_out: 0.3}
    enc_train_output = sess.run(self.enc_train_output, feed_dict=feed_dict)
    return enc_train_output

  def enc_infer_step(self, sess, batch_dict):
    """Single step encoder inference"""
    max_len = batch_dict["enc_inputs"].shape[1]
    feed_dict = { self.enc_inputs: batch_dict["enc_inputs"],
                  self.enc_lens: batch_dict["enc_lens"],
                  self.max_len: max_len,
                  self.drop_out: 0.}
    if(self.bow_pred_method == "seq2seq"):
      feed_dict[self.enc_targets] = batch_dict["enc_targets"]

    enc_output = sess.run(self.enc_infer_output, feed_dict=feed_dict)
    return enc_output

  def dec_infer_step(self, dec_bow, dec_bow_len, max_len, sess):
    """Single step decoder inference"""
    feed_dict = { self.dec_bow: dec_bow,
                  self.dec_bow_len: dec_bow_len,
                  self.max_len: max_len,
                  self.drop_out: 0.}
    dec_output = sess.run(self.dec_infer_output, feed_dict=feed_dict)
    return dec_output

  def dec_train_step(self, sess, batch_dict):
    """Single step training on decoder"""
    max_len = batch_dict["enc_inputs"].shape[1]
    feed_dict = { self.dec_bow: batch_dict["dec_bow"],
                  self.dec_bow_len: batch_dict["dec_bow_len"],
                  self.dec_inputs: batch_dict["dec_inputs"],
                  self.dec_targets: batch_dict["dec_targets"],
                  self.dec_lens: batch_dict["dec_lens"],
                  self.max_len: max_len,
                  self.drop_out: 0.3}
    dec_train_output = sess.run(self.dec_train_output, feed_dict=feed_dict)
    return dec_train_output
  
  def predict(self, sess, batch_dict):
    """predict the sentences and the bow"""
    is_cheat = self.is_cheat
    stop_words = self.stop_words
    batch_size = batch_dict["enc_inputs"].shape[0]
    max_len = batch_dict["enc_inputs"].shape[1]
    output_dict = dict()
    sample_size = self.sample_size

    # Sample from topk
    # dec_bow = _sample_topk(enc_output["pred_ind"], enc_output["pred_prob"])

    if(is_cheat):
      # Sample from golden label
      # golden_prob = _golden_bow_prob(batch_dict["golden_bow"], self.pad_id)
      # dec_bow = _sample_topk_with_enc(batch_dict["golden_bow"], 
      #                                 golden_prob, 
      #                                 batch_dict["enc_inputs"],
      #                                 sample_size, stop_words,
      #                                 self.source_sample_ratio)
      
      dec_bow = batch_dict["dec_bow"]
      dec_bow_len = batch_dict["dec_bow_len"]

      output_dict["pred_ind"] = batch_dict["golden_bow"]
    else:
      # Sample from topk and encoder input
      enc_output = self.enc_infer_step(sess, batch_dict)
      output_dict.update(enc_output)
      dec_bow = _sample_topk_with_enc(enc_output["pred_ind_dec"], 
                                      enc_output["pred_prob_dec"], 
                                      batch_dict["enc_inputs"],
                                      sample_size, stop_words,
                                      self.source_sample_ratio)
      dec_bow_len = np.array([sample_size] * batch_size)
    
    # TODO: sample from top confident 
    output_dict["dec_sample_bow"] = dec_bow
    dec_output = self.dec_infer_step(dec_bow, dec_bow_len, max_len, sess)
    output_dict.update(dec_output)
    return output_dict

  def predict_bow(self, sess, batch_dict):
    return

  def sample_predict(self, sess, batch_dict):
    return

## Auxilary functions 

def _sample_topk(batch_ind, batch_prob, sample_size):
  """sampling from the encoder output """
  sample_ind = []
  for bi, bp in zip(batch_ind, batch_prob):
    sample = np.random.choice(a=bi, size=sample_size, replace=False, p=bp)
    sample_ind.append(sample)
  return np.array(sample_ind)

def _sample_topk_with_enc(batch_ind, batch_prob, enc_inputs, sample_size, 
  stop_words, source_sample_ratio):
  """add the content words from the original sentence"""
  enc_sample_size = int(source_sample_ratio * sample_size)
  topk_sample_size = sample_size - enc_sample_size

  # choose the most confident topk of the predict
  # sample from the origin 
  sample_ind = []
  for ei, bi, bp in zip(enc_inputs, batch_ind, batch_prob):
    enc_bow = list(set(list(ei)) - stop_words)
    if(len(enc_bow) < enc_sample_size):
      enc_sample_size = len(enc_bow)
      topk_sample_size = sample_size - enc_sample_size
    enc_sample = np.random.choice(
      a=enc_bow, size=enc_sample_size, replace=False)

    topk_sample = np.random.choice(
      a=bi, size=topk_sample_size, replace=False, p=bp)
    
    sample = list(enc_sample) + list(topk_sample)
    sample_ind.append(sample)
  return np.array(sample_ind)

def _golden_bow_prob(golden_bow, pad_id):
  probs = []
  for gb in golden_bow:
    p = []
    for w in gb:
      if(w == pad_id): p.append(0)
      else: p.append(1)
    p = np.array(p) / np.sum(p)
    probs.append(p)
  return np.array(probs)

def _sample_confident(pred_ind, pred_prob, confidence_prob):
  """Sample from confident prediction """
  conf_ind, conf_prob = []
  # TBC
  return
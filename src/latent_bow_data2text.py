"""The latent bag of words model for data to text generation 

Yao Fu, Columbia University 
yao.fu@columbia.edu
WED MAY 01ST 2019 
"""

import tensorflow as tf 
import numpy as np 

from tensorflow.nn.rnn_cell import LSTMStateTuple
from tensorflow.contrib import slim
from seq2seq import decoding_infer, decoding_train, attention, create_cell
from bow_seq2seq import (bow_predict_seq_tag, _enc_target_list_to_khot, 
  bow_train_monitor, enc_loss_fn)
from decoder import decode 

################################################################################
## Auxiliary functions

def bow_gumbel_topk_sampling(bow_topk_prob, embedding_matrix, sample_size, 
  vocab_size):
  """Given the soft `bow_topk_prob` k_hot vector, sample `sample_size` locations 
  from it, build the soft memory one the fly"""
  # Not differentiable here 
  prob, ind = tf.nn.top_k(bow_topk_prob, sample_size) # [B, sample_size]
  ind_one_hot = tf.one_hot(ind, vocab_size) # [B, sample_size, V]

  # Differentiable below 
  # [B, 1, V]
  bow_topk_prob_ = tf.expand_dims(bow_topk_prob, [1]) 
  # [B, sample_size, V] -> [B, sample_size]
  sample_prob = tf.reduce_sum(bow_topk_prob_ * ind_one_hot, 2) 
  # [B, sample_size, S]
  sample_memory = tf.nn.embedding_lookup(embedding_matrix, ind) 
  sample_memory *= tf.expand_dims(sample_prob, [2])

  return ind, sample_prob, sample_memory

def _calculate_dec_out_mem_ratio(
  dec_outputs, sample_ind, vocab_size, pad_id, start_id, end_id):
  """Calculate what portion of the output is in the memory"""
  # dec_outputs.shape = [B, T]
  dec_outputs_bow = tf.one_hot(dec_outputs, vocab_size, dtype=tf.float32)
  dec_outputs_bow = tf.reduce_sum(dec_outputs_bow, 1) # [B, V]
  mask = tf.one_hot([start_id, end_id, pad_id], vocab_size, dtype=tf.float32)
  mask = 1. - tf.reduce_sum(mask, 0) # [V]
  dec_outputs_bow *= tf.expand_dims(mask, [0]) 

  sample_ind = tf.one_hot(sample_ind, vocab_size, dtype=tf.float32) # [B, M, V]
  sample_ind = tf.reduce_sum(sample_ind, 1) # [B, V]

  overlap = tf.reduce_sum(dec_outputs_bow * sample_ind, 1) # [B]
  dec_output_support = tf.reduce_sum(dec_outputs_bow, 1) # [B]
  ratio = overlap / dec_output_support

  dec_out_mem_ratio = { 
    "words_from_mem": tf.reduce_mean(overlap),
    "dec_output_bow_cnt": tf.reduce_mean(dec_output_support), 
    "dec_mem_ratio": tf.reduce_mean(ratio)}
  return dec_out_mem_ratio

def _copy_loss(dec_prob_train, dec_targets, dec_mask):
  """"""
  vocab_size = tf.shape(dec_prob_train)[2]
  targets_dist = tf.one_hot(dec_targets, vocab_size)
  loss = tf.reduce_sum(- targets_dist * tf.log(dec_prob_train + 1e-10), 2)
  loss *= dec_mask
  loss = tf.reduce_sum(loss) / tf.reduce_sum(dec_mask)
  return loss 

################################################################################
## Model class 

class LatentBowData2text(object):
  """The latent bow model
  
  The encoder will encode the souce into b and z: 
    b = bow model, regularized by the bow loss
    z = content model

  Then we sample from b with gumbel topk, and construct a dynamic memory on the 
  fly with the sample. The decoder will be conditioned on this memory 
  """

  def __init__(self, config):
    """Initialization"""
    self.mode = config.model_mode
    self.model_name = config.model_name
    self.vocab_size = config.vocab_size
    self.key_size = config.key_size
    self.is_gumbel = config.is_gumbel
    self.gumbel_tau_config = config.gumbel_tau
    self.max_enc_bow = config.max_enc_bow
    self.sample_size = config.sample_size
    self.source_sample_ratio = config.source_sample_ratio
    self.bow_pred_method = config.bow_pred_method
    self.state_size = config.state_size
    self.enc_layers = config.enc_layers
    self.learning_rate = config.learning_rate
    self.learning_rate_enc = config.learning_rate_enc
    self.learning_rate_dec = config.learning_rate_dec
    self.drop_out_config = config.drop_out
    self.optimizer = config.optimizer
    self.dec_start_id = config.dec_start_id
    self.dec_end_id = config.dec_end_id
    self.pad_id = config.pad_id
    self.is_attn = config.is_attn
    self.source_attn = config.source_attn
    self.stop_words = config.stop_words
    self.bow_loss_fn = config.bow_loss_fn
    self.sampling_method = config.sampling_method
    self.topk_sampling_size = config.topk_sampling_size
    self.lambda_enc_loss = config.lambda_enc_loss
    self.no_residual = config.no_residual
    self.copy = config.copy
    self.bow_cond = config.bow_cond
    self.bow_cond_gate = config.bow_cond_gate
    self.num_pointers = config.num_pointers
    return 

  def build(self):
    """Build the model"""
    print("Building the Latent BOW - sequence to sequence model ... ")

    vocab_size = self.vocab_size
    key_size = self.key_size
    state_size = self.state_size
    enc_layers = self.enc_layers
    max_enc_bow = self.max_enc_bow
    lambda_enc_loss = self.lambda_enc_loss

    # Placeholders
    with tf.name_scope("placeholders"):
      enc_keys = tf.placeholder(tf.int32, [None, None], "enc_keys")
      enc_locs = tf.placeholder(tf.int32, [None, None], "enc_locs")
      enc_vals = tf.placeholder(tf.int32, [None, None], "enc_vals")
      enc_lens = tf.placeholder(tf.int32, [None], "enc_lens")
      self.drop_out = tf.placeholder(tf.float32, (), "drop_out")
      self.gumbel_tau = tf.placeholder(tf.float32, (), "gumbel_tau")

      self.enc_keys = enc_keys
      self.enc_locs = enc_locs
      self.enc_vals = enc_vals
      self.enc_lens = enc_lens

      enc_targets = tf.placeholder(tf.int32, [None, None], "enc_targets")
      dec_inputs = tf.placeholder(tf.int32, [None, None], "dec_inputs")
      dec_targets = tf.placeholder(tf.int32, [None, None], "dec_targets")
      dec_lens = tf.placeholder(tf.int32, [None], "dec_lens")

      self.enc_targets = enc_targets
      self.dec_inputs = dec_inputs
      self.dec_targets = dec_targets
      self.dec_lens = dec_lens

    batch_size = tf.shape(enc_keys)[0]
    max_enc_len = tf.shape(enc_keys)[1]
    max_dec_len = tf.shape(dec_targets)[1]

    # Embedding 
    with tf.variable_scope("embeddings"):
      embedding_matrix_vals = tf.get_variable(
        name="embedding_matrix_vals", 
        shape=[vocab_size, state_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=0.05))
      embedding_matrix_keys = tf.get_variable(
        name="embedding_matrix_keys", 
        shape=[key_size, state_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=0.05))
      embedding_matrix_locs = tf.get_variable(
        name="embedding_matrix_locs", 
        shape=[100, state_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=0.05))

      enc_keys = tf.nn.embedding_lookup(embedding_matrix_keys, enc_keys)
      enc_vals = tf.nn.embedding_lookup(embedding_matrix_vals, enc_vals)
      enc_locs = tf.nn.embedding_lookup(embedding_matrix_locs, enc_locs)
      enc_inputs = (enc_keys + enc_vals + enc_locs) / 3.
      dec_inputs = tf.nn.embedding_lookup(embedding_matrix_vals, dec_inputs)

    # Encoder
    with tf.variable_scope("encoder"):
      # TODO: residual LSTM, layer normalization
      enc_cell = [create_cell(
        "enc-%d" % i, state_size, self.drop_out, self.no_residual) 
        for i in range(enc_layers)]
      enc_cell = tf.nn.rnn_cell.MultiRNNCell(enc_cell)
      enc_outputs, enc_state = tf.nn.dynamic_rnn(enc_cell, enc_inputs,
        sequence_length=enc_lens, dtype=tf.float32)

    # Encoder bow prediction
    with tf.variable_scope("bow_output"):
      bow_topk_prob, gumbel_topk_prob, seq_neighbor_ind, seq_neighbor_prob = \
        bow_predict_seq_tag(vocab_size, batch_size, enc_outputs, enc_lens, 
        max_enc_len, self.is_gumbel, self.gumbel_tau)
      seq_neighbor_output = {"seq_neighbor_ind": seq_neighbor_ind, 
        "seq_neighbor_prob": seq_neighbor_prob}
  
    # Encoder output, loss and metrics 
    with tf.name_scope("enc_output"):
      # top k prediction 
      bow_pred_prob, pred_ind = tf.nn.top_k(bow_topk_prob, max_enc_bow)

      # loss function 
      enc_targets = _enc_target_list_to_khot(
        enc_targets, vocab_size, self.pad_id)
      enc_loss = enc_loss_fn(
        self.bow_loss_fn, enc_targets, bow_topk_prob, max_enc_bow)
      self.train_output = {"enc_loss": enc_loss}

      # performance monitor 
      bow_metrics_dict = bow_train_monitor(
        bow_topk_prob, pred_ind, vocab_size, batch_size, enc_targets)
      self.train_output.update(bow_metrics_dict)

    # Encoder soft sampling 
    with tf.name_scope("gumbel_topk_sampling"):
      sample_ind, sample_prob, sample_memory = bow_gumbel_topk_sampling(
        gumbel_topk_prob, embedding_matrix_vals, self.sample_size, vocab_size)
      sample_memory_lens = tf.ones(batch_size, tf.int32) * self.sample_size
      sample_memory_avg = tf.reduce_mean(sample_memory, 1) # [B, S]

      sample_memory_output = {"bow_pred_ind": pred_ind, 
                              "bow_pred_prob": bow_pred_prob, 
                              "sample_memory_ind": sample_ind, 
                              "sample_memory_prob": sample_prob }

    # Decoder 
    # The initial state of the decoder = 
    #   encoder meaning vector z + encoder bow vector b 
    with tf.variable_scope("decoder"):
      dec_cell = [create_cell(
        "dec-%d" % i, state_size, self.drop_out, self.no_residual) 
        for i in range(enc_layers)]
      dec_cell = tf.nn.rnn_cell.MultiRNNCell(dec_cell)
      dec_proj = tf.layers.Dense(vocab_size, name="dec_proj",
        kernel_initializer=tf.random_normal_initializer(stddev=0.05),
        bias_initializer=tf.constant_initializer(0.))
      dec_ptr_k_proj = [
        tf.layers.Dense(state_size, name="dec_ptr_k_proj_%d" % pi,
        kernel_initializer=tf.random_normal_initializer(stddev=0.05),
        bias_initializer=tf.constant_initializer(0.)) 
        for pi in range(self.num_pointers)]
      dec_ptr_g_proj = tf.layers.Dense(1, name="dec_ptr_g_proj",
        kernel_initializer=tf.random_normal_initializer(stddev=0.05),
        bias_initializer=tf.constant_initializer(0.),
        activation=tf.nn.sigmoid)
      bow_cond_gate_proj = tf.layers.Dense(1, name="bow_cond_gate_proj",
        kernel_initializer=tf.random_normal_initializer(stddev=0.05),
        bias_initializer=tf.constant_initializer(0.),
        activation=tf.nn.sigmoid)

      dec_init_state = []
      for l in range(enc_layers):
        dec_init_state.append(LSTMStateTuple(c=enc_state[0].c, 
                                h=enc_state[0].h + sample_memory_avg))
      dec_init_state = tuple(dec_init_state)

      # if(enc_layers == 2):
      #   dec_init_state = (LSTMStateTuple( c=enc_state[0].c, 
      #                                     h=enc_state[0].h + sample_memory_avg),
      #                     LSTMStateTuple( c=enc_state[1].c, 
      #                                     h=enc_state[1].h + sample_memory_avg) )
      # elif(enc_layers == 4):
      #   dec_init_state = (LSTMStateTuple(c=enc_state[0].c, 
      #                       h=enc_state[0].h + sample_memory_avg),
      #                     LSTMStateTuple( c=enc_state[1].c, 
      #                       h=enc_state[1].h + sample_memory_avg) )
      # else: raise Exception('enc_layers not in [2, 4]')

      if(self.source_attn):
        # [B, M + T, S]
        dec_memory = [sample_memory, enc_outputs]
        dec_mem_len = [sample_memory_lens, enc_lens]
        dec_max_mem_len = [self.sample_size, max_enc_len]
      else:
        dec_memory = sample_memory
        dec_mem_len = sample_memory_lens
        dec_max_mem_len = tf.shape(dec_memory)[1] 

      if(self.bow_cond): bow_cond = sample_memory_avg
      else: bow_cond = None

      if(self.bow_cond_gate == False): bow_cond_gate_proj = None

      (dec_outputs_predict, dec_logits_train, dec_prob_train, pointer_ent, 
        avg_max_ptr, avg_num_copy) = decode( 
        self.dec_start_id, dec_inputs, 
        dec_cell, dec_proj, embedding_matrix_vals, 
        dec_init_state, dec_memory, dec_mem_len, dec_max_mem_len, 
        batch_size, max_dec_len, self.sampling_method, self.topk_sampling_size,
        state_size, multi_source=True, copy=self.copy, copy_ind=sample_ind,
        dec_ptr_g_proj=dec_ptr_g_proj, dec_ptr_k_proj=dec_ptr_k_proj,
        bow_cond=bow_cond, bow_cond_gate_proj=bow_cond_gate_proj)

    # model saver, before the optimizer 
    all_variables = slim.get_variables_to_restore()
    model_variables = [var for var in all_variables 
      if var.name.split("/")[0] == self.model_name]
    print("%s model, variable list:" % self.model_name)
    for v in model_variables: print("  %s" % v.name)
    self.model_saver = tf.train.Saver(model_variables, max_to_keep=3)

    with tf.variable_scope("optimizer"):
      optimizer = tf.train.AdamOptimizer(self.learning_rate)

    # decoder output, training and inference, combined with encoder loss 
    with tf.name_scope("dec_output"):
      dec_mask = tf.sequence_mask(dec_lens, max_dec_len, dtype=tf.float32)
      if(self.copy == False):
        dec_loss = tf.contrib.seq2seq.sequence_loss(
          dec_logits_train, dec_targets, dec_mask)
      else: 
        dec_loss = _copy_loss(dec_prob_train, dec_targets, dec_mask)

      loss = dec_loss + lambda_enc_loss * enc_loss
      train_op = optimizer.minimize(loss)

      dec_output = {"train_op": train_op, "dec_loss": dec_loss, "loss": loss}
      self.train_output.update(dec_output)
      if(self.copy):
        pointer_ent =\
          tf.reduce_sum(pointer_ent * dec_mask) / tf.reduce_sum(dec_mask)
        self.train_output['pointer_ent'] = pointer_ent
        avg_max_ptr =\
          tf.reduce_sum(avg_max_ptr * dec_mask) / tf.reduce_sum(dec_mask)
        self.train_output['avg_max_ptr'] = avg_max_ptr
        avg_num_copy = tf.reduce_sum(avg_num_copy * dec_mask, 1)
        avg_num_copy = tf.reduce_mean(avg_num_copy)
        self.train_output['avg_num_copy'] = avg_num_copy

      self.infer_output = {"dec_predict": dec_outputs_predict}
      dec_out_mem_ratio = _calculate_dec_out_mem_ratio(dec_outputs_predict, 
        sample_ind, vocab_size, self.pad_id, self.dec_start_id, self.dec_end_id)
      self.infer_output.update(dec_out_mem_ratio)
      self.infer_output.update(sample_memory_output)
      self.infer_output.update(seq_neighbor_output)
    return 

  def train_step(self, sess, batch_dict, ei):
    """Single step training"""
    feed_dict = { self.enc_keys: batch_dict["enc_keys"], 
                  self.enc_vals: batch_dict["enc_vals"],
                  self.enc_locs: batch_dict["enc_locs"], 
                  self.enc_lens: batch_dict["enc_lens"],
                  self.enc_targets: batch_dict["dec_bow"],
                  self.dec_inputs: batch_dict["dec_inputs"],
                  self.dec_targets: batch_dict["dec_targets"],
                  self.dec_lens: batch_dict["dec_lens"],
                  self.drop_out: self.drop_out_config,
                  self.gumbel_tau: self.gumbel_tau_config}
    output_dict = sess.run(self.train_output, feed_dict=feed_dict)
    return output_dict

  def predict(self, sess, batch_dict):
    """Single step prediction"""
    feed_dict = { self.enc_keys: batch_dict["enc_keys"], 
                  self.enc_vals: batch_dict["enc_vals"],
                  self.enc_locs: batch_dict["enc_locs"], 
                  self.enc_lens: batch_dict["enc_lens"],
                  self.dec_targets: batch_dict['dec_targets'],
                  self.drop_out: 0.,
                  # self.gumbel_tau: self.gumbel_tau_config, # soft sample 
                  self.gumbel_tau: 0.00001 # near-hard sample
                  }
    output_dict = sess.run(self.infer_output, feed_dict=feed_dict)
    return output_dict
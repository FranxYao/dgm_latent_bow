"""The language model, used for evaluation

Yao Fu, Columbia University
yao.fu@columbia.edu
APR 11ST 2019 
"""

import tensorflow as tf 
import numpy as np 

from tensorflow.contrib import slim
from seq2seq import create_cell

class LM(object):
  """The language model, used for evaluation"""

  def __init__(self, config):
    self.vocab_size = config.vocab_size
    self.state_size = config.state_size
    self.enc_layers = config.enc_layers
    self.learning_rate = config.learning_rate
    self.optimizer = config.optimizer
    self.config_drop_out = config.drop_out
    return 

  def build(self):
    print("Building the language model ... ")

    vocab_size = self.vocab_size
    state_size = self.state_size
    enc_layers = self.enc_layers

    with tf.name_scope("placeholders"):
      enc_inputs = tf.placeholder(tf.int32, [None, None], "enc_inputs")
      targets = tf.placeholder(tf.int32, [None, None], "targets")
      inp_lens = tf.placeholder(tf.int32, [None], "inp_lens")
      self.drop_out = tf.placeholder(tf.float32, (), "drop_out")

      self.enc_inputs = enc_inputs
      self.inp_lens = inp_lens
      self.targets = targets

    batch_size = tf.shape(enc_inputs)[0]
    max_len = tf.shape(enc_inputs)[1]

    with tf.variable_scope("embeddings"):
      embedding_matrix = tf.get_variable(
        "embedding_matrix", [vocab_size, state_size])
      enc_inputs = tf.nn.embedding_lookup(embedding_matrix, enc_inputs)

    with tf.variable_scope("encoder"):
      # TODO: residual LSTM, layer normalization
      enc_cell = [create_cell("enc-%d" % i, state_size, self.drop_out) 
        for i in range(enc_layers)]
      enc_cell = tf.nn.rnn_cell.MultiRNNCell(enc_cell)
      enc_outputs, enc_state = tf.nn.dynamic_rnn(enc_cell, enc_inputs,
        sequence_length=inp_lens, dtype=tf.float32)

      enc_proj = tf.layers.Dense(vocab_size, name="enc_proj")
      enc_logits = enc_proj(enc_outputs)

      mask = tf.sequence_mask(inp_lens, max_len, dtype=tf.float32)
      loss = tf.contrib.seq2seq.sequence_loss(
          enc_logits, targets, mask)

      # get variables before optimizer 
      all_variables = slim.get_variables_to_restore()
      lm_variables = [var for var in all_variables if var.name[:2] == "lm"]
      print("lm model, variable list:")
      for v in lm_variables: print("  %s" % v.name)
      self.model_saver = tf.train.Saver(lm_variables, max_to_keep=10)

      optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
      train_op = optimizer.minimize(loss)

      self.train_output = { "train_op": train_op, 
                            "loss": loss,
                            "ppl": tf.exp(loss)}
      self.eval_output = {"loss": loss,
                          "ppl": tf.exp(loss)}

      
    return 

  def train_step(self, sess, batch_dict, ei):
    """One step training"""
    feed_dict = { self.enc_inputs: batch_dict["dec_inputs"],
                  self.targets: batch_dict["targets"], 
                  self.inp_lens: batch_dict["out_lens"],
                  self.drop_out: self.config_drop_out}
    output_dict = sess.run(self.train_output, feed_dict = feed_dict)
    return output_dict

  def eval_step(self, sess, batch_dict):
    """One step evaluation"""
    feed_dict = { self.enc_inputs: batch_dict["dec_inputs"], 
                  self.targets: batch_dict["targets"], 
                  self.inp_lens: batch_dict["out_lens"],
                  self.drop_out: 0.}
    output_dict = sess.run(self.eval_output, feed_dict = feed_dict)
    return output_dict

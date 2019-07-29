"""The hierarchical vae model

Yao Fu, Columbia University
yao.fu@columbia.edu
SAT APR 20TH 2019
"""

import tensorflow as tf 

def kl_divergence():
  """The KL divergence term of the hierarchical model

  The KL divergence of this model does not have closed form solution (or I am 
  just too lazy to calculate it). So we use Monte Carlo estimation 
  """
  return 

class HierarchicalVAE(object):
  """The hierarchical VAE model"""

  def __init__(self, config):
    return

  def build(self):
    """Build the model"""
    print("Building the hierarchical VAE model ... ")

    vocab_size = self.vocab_size
    state_size = self.state_size
    enc_layers = self.enc_layers
    layer_norm = self.layer_norm

    # Placeholders
    with tf.name_scope("placeholders"):
      # enc_inputs.shape = [batch_size, num_paraphrases, max_sent_len]
      enc_inputs = tf.placeholder(tf.int32, [None, None, None], "enc_inputs")
      # inp_lens.shape = [batch_size, num_paraphrases]
      inp_lens = tf.placeholder(tf.int32, [None, None], "inp_lens")
      self.drop_out = tf.placeholder(tf.float32, (), "drop_out")

      self.enc_inputs = enc_inputs
      self.inp_lens = inp_lens

      if(self.mode == "train"):
        dec_inputs = tf.placeholder(tf.int32, [None, None, None], "dec_inputs")
        targets = tf.placeholder(tf.int32, [None, None, None], "targets")
        out_lens = tf.placeholder(tf.int32, [None, None], "out_lens")

        self.dec_inputs = dec_inputs
        self.targets = targets
        self.out_lens = out_lens

    batch_size = tf.shape(enc_inputs)[0]
    num_paraphrases = tf.shape(enc_inputs)[1]
    max_len = tf.shape(enc_inputs)[2]

    # Embeddings 
    with tf.variable_scope("embeddings"):
      embedding_matrix = tf.get_variable(
        "embedding_matrix", [vocab_size, state_size])
      enc_inputs = tf.nn.embedding_lookup(embedding_matrix, enc_inputs)

      if(self.mode == "train"): 
        dec_inputs = tf.nn.embedding_lookup(embedding_matrix, dec_inputs)

    # Encoder
    with tf.variable_scope("encoder"):
      # lstm encoder
      enc_cell = [create_cell("enc-%d" % i, state_size, self.drop_out) 
        for i in range(enc_layers)]
      enc_cell = tf.nn.rnn_cell.MultiRNNCell(enc_cell)
      enc_inputs = tf.reshape(
        enc_inputs, [batch_size * num_paraphrases, max_len, state_size])
      enc_outputs, enc_state = tf.nn.dynamic_rnn(enc_cell, enc_inputs,
        sequence_length=inp_lens, dtype=tf.float32)

      if(layer_norm): 
        enc_outputs = tf.contrib.layers.layer_norm(enc_outputs)

      # inference network


    return

  def train_step(self, sess, batch_dict):
    return

  def valid_step(self, sess, batch_dict):
    return

  def predict_step(self, sess, batch_dict):
    return


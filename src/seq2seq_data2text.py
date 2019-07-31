"""The baseline sequence to sequence model 

Yao Fu, Columbia University
yao.fu@columabi.edu
APR 09TH 2019
"""

import tensorflow as tf 
import numpy as np 
import tensorflow.contrib.distributions as tfd
from tensorflow.nn.rnn_cell import LSTMStateTuple

from tensorflow.contrib import slim
from decoder import decoding_infer, decoding_train, attention

def create_cell(name, state_size, drop_out, no_residual=False):
  """Create a LSTM cell"""
  # This one should be the fastest
  cell = tf.nn.rnn_cell.LSTMCell(state_size, name=name, dtype=tf.float32,
    initializer=tf.random_normal_initializer(stddev=0.05))
  cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1. - drop_out)
  if(no_residual == False): 
    print('use residual ... ')
    cell = tf.nn.rnn_cell.ResidualWrapper(cell)
  else: print('not use residual')
  return cell

class Seq2seqData2text(object):
  """The sequence to sequence model"""

  def __init__(self, config):
    self.mode = config.model_mode
    self.model_name = config.model_name
    self.vocab_size = config.vocab_size
    self.state_size = config.state_size
    self.enc_layers = config.enc_layers
    self.init_learning_rate = config.learning_rate
    self.optimizer = config.optimizer
    self.dec_start_id = config.dec_start_id
    self.dec_end_id = config.dec_end_id
    self.is_attn = config.is_attn
    self.decoding_mode = config.decoding_mode
    self.sampling_method = config.sampling_method
    self.topk_sampling_size = config.topk_sampling_size
    self.learning_rate_decay = config.learning_rate_decay
    self.no_residual = config.no_residual

    self.vae = config.vae_seq2seq
    self.lambda_kl_config = config.lambda_kl
    self.prior = config.prior

    # debug
    self.inspect = {}
    return 

  def build(self):
    """Build the model"""
    print("Building the seq2seq model for data to text generation... ")

    vocab_size = self.vocab_size
    state_size = self.state_size
    enc_layers = self.enc_layers

    # Placeholders
    with tf.name_scope("placeholders"):
      enc_keys = tf.placeholder(tf.int32, [None, None], "enc_keys")
      enc_locs = tf.placeholder(tf.int32, [None, None], "enc_locs")
      enc_vals = tf.placeholder(tf.int32, [None, None], "enc_vals")
      enc_lens = tf.placeholder(tf.int32, [None], "enc_lens")
      self.drop_out = tf.placeholder(tf.float32, (), "drop_out")

      self.enc_keys = enc_keys
      self.enc_locs = enc_locs
      self.enc_vals = enc_vals
      self.enc_lens = enc_lens

      dec_inputs = tf.placeholder(tf.int32, [None, None], "dec_inputs")
      dec_targets = tf.placeholder(tf.int32, [None, None], "dec_targets")
      dec_lens = tf.placeholder(tf.int32, [None], "dec_lens")
      self.learning_rate = tf.placeholder(tf.float32, (), "learning_rate")
      self.lambda_kl = tf.placeholder(tf.float32, (), "lambda_kl")

      self.dec_inputs = dec_inputs
      self.dec_targets = dec_targets
      self.dec_lens = dec_lens

    batch_size = tf.shape(enc_keys)[0]
    max_enc_len = tf.shape(enc_keys)[1]
    max_dec_len = tf.shape(dec_inputs)[1]

    # Embedding 
    with tf.variable_scope("embeddings"):
      embedding_matrix_vals = tf.get_variable(
        name="embedding_matrix_vals", 
        shape=[vocab_size, state_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=0.05))
      embedding_matrix_keys = tf.get_variable(
        name="embedding_matrix_keys", 
        shape=[vocab_size, state_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=0.05))
      embedding_matrix_locs = tf.get_variable(
        name="embedding_matrix_locs", 
        shape=[vocab_size, state_size],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(stddev=0.05))

      enc_keys = tf.nn.embedding_lookup(embedding_matrix_keys, enc_keys)
      enc_vals = tf.nn.embedding_lookup(embedding_matrix_vals, enc_vals)
      enc_locs = tf.nn.embedding_lookup(embedding_matrix_locs, enc_locs)

      enc_inputs = (enc_keys + enc_vals + enc_locs) / 3.

      if(self.mode == "train"): 
        dec_inputs = tf.nn.embedding_lookup(embedding_matrix_vals, dec_inputs)

    # Encoder
    with tf.variable_scope("encoder"):
      # TODO: residual LSTM, layer normalization
      # if(self.bidirectional)
      #   enc_cell_fw = [create_cell(
      #     "enc-fw-%d" % i, state_size, self.drop_out, self.no_residual) 
      #     for i in range(enc_layers)]
      #   enc_cell_bw = [create_cell(
      #     "enc-bw-%d" % i, state_size, self.drop_out, self.no_residual) 
      #     for i in range(enc_layers)]
      # else: 
      enc_cell = [create_cell(
        "enc-%d" % i, state_size, self.drop_out, self.no_residual) 
        for i in range(enc_layers)]
      enc_cell = tf.nn.rnn_cell.MultiRNNCell(enc_cell)
      enc_outputs, enc_state = tf.nn.dynamic_rnn(enc_cell, enc_inputs,
        sequence_length=enc_lens, dtype=tf.float32)

    # Decoder 
    with tf.variable_scope("decoder"):
      dec_cell = [create_cell(
        "dec-%d" % i, state_size, self.drop_out, self.no_residual) 
        for i in range(enc_layers)]
      dec_cell = tf.nn.rnn_cell.MultiRNNCell(dec_cell)

      dec_proj = tf.layers.Dense(vocab_size, name="dec_proj",
        kernel_initializer=tf.random_normal_initializer(stddev=0.05),
        bias_initializer=tf.constant_initializer(0.))

    # latent code 
    if(self.vae):
      print("Using vae model")
      with tf.variable_scope("latent_code"):
        enc_mean = tf.reduce_sum(enc_outputs, 1) 
        enc_mean /= tf.expand_dims(tf.cast(enc_lens, tf.float32), [1])
        z_code = enc_mean

        if(self.prior == "gaussian"):
          print("Gaussian prior")
          latent_proj = tf.layers.Dense(2 * state_size, name="latent_proj",
            kernel_initializer=tf.random_normal_initializer(stddev=0.05),
            bias_initializer=tf.constant_initializer(0.))
          z_loc, z_scale = tf.split(
            latent_proj(z_code), [state_size, state_size], 1)
          z_mvn = tfd.MultivariateNormalDiag(z_loc, z_scale)
          z_sample = z_mvn.sample()
        
        elif(self.prior == "vmf"):
          # print("vmf prior")
          # latent_proj = tf.layers.Dense(state_size + 1, name="latent_proj",
          #   kernel_initializer=tf.random_normal_initializer(stddev=0.05),
          #   bias_initializer=tf.constant_initializer(0.))
          # z_mu, z_conc = tf.split(
          #   latent_proj(z_code), [state_size, 1], 1)
          # z_mu /= tf.expand_dims(tf.norm(z_mu, axis=1), axis=1)
          # z_conc = tf.reshape(z_conc, [batch_size])
          # z_vmf = tfd.VonMisesFisher(z_mu, z_conc)
          # z_sample = z_vmf.sample()
          pass

        dec_init_state = (LSTMStateTuple(c=z_sample, h=z_sample),
                          LSTMStateTuple(c=z_sample, h=z_sample))

    else: 
      print("Using normal seq2seq, no latent variable")
      dec_init_state = enc_state

    with tf.variable_scope("decoding"):
      # greedy decoding
      _, dec_outputs_predict = decoding_infer(self.dec_start_id,
                                              dec_cell,
                                              dec_proj,
                                              embedding_matrix_vals,
                                              dec_init_state,
                                              enc_outputs,
                                              batch_size,
                                              max_dec_len,
                                              enc_lens,
                                              max_enc_len,
                                              self.is_attn,
                                              self.sampling_method,
                                              self.topk_sampling_size,
                                              state_size=self.state_size)
      # decoding with forward sampling
      # dec_outputs_sampling = decodeing_infer() #  TBC

      if(self.mode == "train"):
        # training decoding
        dec_logits_train, _, _, _, _ = decoding_train( dec_inputs, 
                                            dec_cell, 
                                            dec_proj,
                                            dec_init_state, 
                                            enc_outputs,  
                                            max_dec_len, 
                                            enc_lens, 
                                            max_enc_len,
                                            self.is_attn,
                                            self.state_size)

        all_variables = slim.get_variables_to_restore()
        model_variables = [var for var in all_variables 
          if var.name.split("/")[0] == self.model_name]
        print("%s model, variable list:" % self.model_name)
        for v in model_variables: print("  %s" % v.name)
        self.model_saver = tf.train.Saver(all_variables, max_to_keep=3)

        # loss and optimizer
        dec_mask = tf.sequence_mask(dec_lens, max_dec_len, dtype=tf.float32)
        dec_loss = tf.contrib.seq2seq.sequence_loss(
          dec_logits_train, dec_targets, dec_mask)

        if(self.vae):
          if(self.prior == "gaussian"):
            standard_normal = tfd.MultivariateNormalDiag(
              tf.zeros(state_size), tf.ones(state_size))

            prior_prob = standard_normal.log_prob(z_sample) # [B] 
            posterior_prob = z_mvn.log_prob(z_sample) # [B]
            kl_loss = tf.reduce_mean(posterior_prob - prior_prob)
            loss = dec_loss + self.lambda_kl * kl_loss

          elif(self.prior == "vmf"):
            # vmf_mu_0 = tf.ones(state_size) / tf.cast(state_size, tf.float32)
            # standard_vmf = tfd.VonMisesFisher(vmf_mu_0, 0)
            # prior_prob = standard_vmf.log_prob(z_sample) # [B] 
            # posterior_prob = z_vmf.log_prob(z_sample) # [B]
            # kl_loss = tf.reduce_mean(posterior_prob - prior_prob)
            # loss = dec_loss + self.lambda_kl * kl_loss
            pass
        else:
          loss = dec_loss

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.minimize(loss)

        self.train_output = { "train_op": train_op, 
                              "loss": loss}
        self.train_output.update(self.inspect)
        if(self.vae):
          self.train_output["dec_loss"] = dec_loss
          self.train_output["kl_loss"] = kl_loss

        self.valid_output = {"nll": tf.exp(loss)}
        self.infer_output = {"dec_predict": dec_outputs_predict}  

      else: 
        self.infer_output = {"dec_predict": dec_outputs_predict}
    return 

  def train_step(self, sess, batch_dict, ei):
    """One step training"""
    # learning rate decay
    if(self.learning_rate_decay):
      lr_decay = float((ei + 3) // 3)
      lr = self.init_learning_rate / lr_decay
    else: lr = self.init_learning_rate
    
    # kl annealing
    # lambda_kl = (ei // 2) * self.lambda_kl_config + 0.001
    lambda_kl = self.lambda_kl_config
    feed_dict = { self.enc_keys: batch_dict["enc_keys"], 
                  self.enc_vals: batch_dict["enc_vals"],
                  self.enc_locs: batch_dict["enc_locs"],  
                  self.enc_lens: batch_dict["enc_lens"],
                  self.dec_inputs: batch_dict["dec_inputs"],
                  self.dec_targets: batch_dict["dec_targets"],
                  self.dec_lens: batch_dict["dec_lens"],
                  self.drop_out: batch_dict["drop_out"],
                  self.learning_rate: lr,
                  self.lambda_kl: lambda_kl}
    output_dict = sess.run(self.train_output, feed_dict=feed_dict)
    return output_dict

  def valid_step(self, sess, batch_dict):
    """Validation, predict NLL, used to evaluate the density estimation 
    performance"""
    feed_dict = { self.enc_keys: batch_dict["enc_keys"], 
                  self.enc_vals: batch_dict["enc_vals"],
                  self.enc_locs: batch_dict["enc_locs"], 
                  self.enc_lens: batch_dict["enc_lens"],
                  self.dec_inputs: batch_dict["dec_inputs"],
                  self.dec_targets: batch_dict["dec_targets"],
                  self.dec_lens: batch_dict["dec_lens"],
                  self.drop_out: 0.}
    output_dict = sess.run(self.valid_output, feed_dict=feed_dict)
    return output_dict

  def predict_greedy(self, sess, batch_dict):
    """Greedy decoding, always choose the word with the highest probability"""
    feed_dict = { self.enc_keys: batch_dict["enc_keys"], 
                  self.enc_vals: batch_dict["enc_vals"],
                  self.enc_locs: batch_dict["enc_locs"], 
                  self.enc_lens: batch_dict["enc_lens"],
                  self.dec_inputs: batch_dict['dec_inputs'],
                  self.drop_out: 0.}
    output_dict = sess.run(self.infer_output, feed_dict=feed_dict)
    return output_dict

  def predict(self, sess, batch_dict):
    """Predict with different mode"""
    if(self.decoding_mode == "greedy"):
      output_dict = self.predict_greedy(sess, batch_dict)
    elif(self.decoding_mode == "random_walk"):
      output_dict = self.random_walk(sess, batch_dict)
    elif(self.decoding_mode == "topk_sampling"):
      output_dict = self.predict_topk_sampling(sess, batch_dict)
    return output_dict

  def random_walk(self, sess, batch_dict, step_length=5):
    """Random walk decoding, start from p0 -> p1 -> p2 -> p3 ... """
    outputs = []
    feed_dict = { self.enc_inputs: batch_dict["enc_inputs"], 
                  self.enc_lens: batch_dict["enc_lens"],
                  self.drop_out: 0.}
    def _feed_dict_update(feed_dict, output_dict):
      batch_size = output_dict["dec_predict"].shape[0]
      inputs = output_dict["dec_predict"].T[1:].T
      inputs = np.concatenate(
        [np.zeros([batch_size, 1]) + self.dec_start_id, inputs], axis=1)

      enc_lens = []
      max_sent_len = inputs.shape[1]
      for s in inputs:
        for l in range(max_sent_len):
          if(s[l] == self.dec_end_id): break
        enc_lens.append(l)

      feed_dict[self.enc_inputs] = inputs
      feed_dict[self.enc_lens] = np.array(enc_lens)
      return feed_dict

    for i in range(step_length):
      output_dict = sess.run(self.infer_output, feed_dict=feed_dict)
      outputs.append(output_dict)
      feed_dict = _feed_dict_update(feed_dict, output_dict)
    
    return outputs

  def predict_topk_sampling(self, sess, batch_dict):
    """Forwar sampling decoding, at certain steps, randomly sample the word with
    top k probability"""
    return

  def predict_beamsearch(self, sess, batch_dict):
    """Beam search decoing"""
    return

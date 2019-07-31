

import os

import tensorflow as tf 
import numpy as np 

from config import Config
from controller import Controller
from data_utils import Dataset
from data_utils_table2text import DatasetTable2text
from seq2seq import Seq2seq
from nlp_pipeline import corpus_statistics

import time

from tqdm import tqdm 


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("controller_mode", None, 
  "training task, 'train' or 'test'")

flags.DEFINE_integer("start_epoch", -1, "The index of the start epoch")

flags.DEFINE_integer("num_epoch", -1, "Number of training epoch")

flags.DEFINE_integer("batch_size", -1, "Size of the batch")
flags.DEFINE_integer("state_size", -1, "Size of the hidden state")
  
flags.DEFINE_integer("train_print_interval", -1, 
  "The print interval during training")

flags.DEFINE_integer("topk_sampling_size", -1, 
  "The print interval during training")
flags.DEFINE_integer("enc_layers", -1, 
  "Number of layers in the encoder")

flags.DEFINE_string("optimizer", "", "The optimizer")
flags.DEFINE_float("learning_rate", -1., "The learning rate")
flags.DEFINE_float("learning_rate_enc", -1., "The learning rate")
flags.DEFINE_float("learning_rate_dec", -1., "The learning rate")
flags.DEFINE_float("drop_out", -1., "Drop out")
flags.DEFINE_float("gumbel_tau", 0.5, "Gumbel temperature")
flags.DEFINE_boolean("is_attn", True, "use attention in sequence to sequence")
flags.DEFINE_boolean("is_gumbel", False, "use gumbel sampling")
flags.DEFINE_boolean("vae_seq2seq", False, "use vae model")
flags.DEFINE_boolean("save_ckpt", False, "Whether saving the model")
flags.DEFINE_boolean("is_cheat", False, "Whether the decoder should cheat")
flags.DEFINE_boolean("no_residual", False, 'if not use residual LSTM')
flags.DEFINE_boolean("single_ref", False, "Whether to use single reference")
flags.DEFINE_boolean("copy", False, "Whether to use copy mechanism")
flags.DEFINE_integer("num_pointers", -1, "The number of pointers in the copy mode")
flags.DEFINE_boolean("full_quora", False, "Whether to use the full quora dataset")
flags.DEFINE_boolean("bow_cond", False, "Whether to use the bow as the decoder condition ")
flags.DEFINE_boolean("bow_cond_gate", False, "Whether to use gete for the bow condition vector")
flags.DEFINE_float("lambda_kl", -1, "lambda of kl loss term")
flags.DEFINE_string("prior", "", "The prior for the vae model")
flags.DEFINE_string("bow_loss", "", "The bag of word loss")

flags.DEFINE_string("dataset", "", "The dataset to use")

flags.DEFINE_string("model_name", "", "The name of the model")

flags.DEFINE_string("model_version", "", "The version of the model")

flags.DEFINE_string("gpu_id", "", "gpu index")

FLAGS.gpu_id = "0"
FLAGS.model_version = "test"

config = Config()
config.parse_arg(FLAGS)
config.print_arg()

# dataset
# dset = Dataset(config)
dset = DatasetTable2text(config)
# dset.build()
dset.load()
batch = dset.next_batch('train', 100)
dset.print_predict(batch_dict=batch, num_cases=3) 

enc_vals = []

# batch = next_batch(dset, 'train', 100)
# dset.print_predict(batch_dict=batch, num_cases=3)
# print_predict(dset, batch_dict=batch, num_cases=3)
num_batches = dset.num_batches(100, 'train')
for i in tqdm(range(num_batches)):
  batch = dset.next_batch('train', 100)
  # enc_vals.append(batch['enc_vals'])
  # next_batch(dset, 'train', 100)
enc_vals = np.stack(enc_vals[:5800])
np.save('enc_vals', enc_vals)
dset.print_predict(batch_dict=batch, num_cases=3) 

# print('train %d batches' % num_batches)
# num_batches = dset.num_batches(100, 'dev')
# print('dev %d batches' % num_batches)


# def print_sent(s, id2word):
#   s = ' '.join([id2word[i] for i in s])
#   return s 

# sents, vals, keys, pos, _ = dset._dataset['dev']
# sent_set = [[s] for s in sents]
# val_set = [[v] for v in vals]
# corpus_statistics(sent_set)
# corpus_statistics(val_set)

# print_sent(sents[0], dset.id2word)
# print_sent(vals[0], dset.id2word)
# print_sent(keys[0], dset.id2key)

# dset.build()
# config.vocab_size = dset.vocab_size
# config.dec_start_id = dset.word2id["_GOO"]
# config.dec_end_id = dset.word2id["_EOS"]

# # model 
# Model = Seq2seq
# model = Model(config)
# model.build()

# # training 
# start_epoch = config.start_epoch
# num_epoch = config.num_epoch
# batch_size = config.batch_size
# model_name = config.model_name
# drop_out = config.drop_out
# print_interval = config.train_print_interval

# os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id

# gpu_config = tf.ConfigProto()
# gpu_config.gpu_options.allow_growth = True
# sess = tf.Session(config=gpu_config)
# sess.run(tf.global_variables_initializer())

# num_batches = dset.num_batches(batch_size, "train")
# print("%d batches in total" % num_batches)
# ei = 0
# start_time = time.time()
# loss = []

# for bi in range(10):
#   batch_dict = dset.next_batch("train", batch_size, model_name)
#   batch_dict["drop_out"] = drop_out
#   output_dict = model.train_step(sess, batch_dict)

#   loss.append(output_dict["loss"])

#   if(bi % print_interval == 0):
#     print("e-%d, b-%d, t-%.2f, l-%.4f" % 
#       (ei, bi, time.time() - start_time, output_dict["loss"]))

# print("\nepoch %d, time %.2f, average loss %.4f\n" % 
#   (ei, time.time() - start_time, np.average(loss)))

# # predict 
# batch_dict = dset.next_batch("dev", batch_size, "eval")
# batch_dict["drop_out"] = 0.
# output_dict = model.predict_greedy(sess, batch_dict)
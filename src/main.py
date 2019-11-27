"""The paraphrase generation model main

Yao Fu, Columbia University
yao.fu@columbia.edu
APR 10TH 2019
"""

import tensorflow as tf 
import numpy as np 

from config import Config
from controller import Controller
from data_utils import Dataset
from data_utils_table2text import DatasetTable2text
from seq2seq import Seq2seq
from bow_seq2seq import BowSeq2seq
from latent_bow import LatentBow
from latent_bow_data2text import LatentBowData2text
from seq2seq_data2text import Seq2seqData2text
from lm import LM

from tqdm import tqdm
from time import time

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

def main():
  # configuration
  config = Config()
  config.parse_arg(FLAGS)
  config.setup_path()
  config.print_arg()

  # dataset
  if(config.dataset == 'wikibio'):
    dset = DatasetTable2text(config)
    dset.load()
    config.key_size = len(dset.key2id)
  else: 
    dset = Dataset(config)
    dset.build()
  config.vocab_size = len(dset.word2id)
  config.dec_start_id = dset.word2id["_GOO"]
  config.dec_end_id = dset.word2id["_EOS"]
  config.pad_id = dset.pad_id
  config.stop_words = dset.stop_words

  # model 
  if(config.model_name == "seq2seq"): 
    if(config.dataset == 'wikibio'): Model = Seq2seqData2text
    else: Model = Seq2seq
  elif(config.model_name == "bow_seq2seq"): Model = BowSeq2seq
  elif(config.model_name == "vae"): Model = Vae
  elif(config.model_name == "hierarchical_vae"): Model = Hierarchical_Vae
  elif(config.model_name == "latent_bow"): 
    if(config.dataset == 'wikibio'): Model = LatentBowData2text
    else: Model = LatentBow
  elif(config.model_name == "lm"): Model = LM
  else: 
    msg = "the model name shoule be in ['seq2seq', 'vae', 'hierarchical_vae', 'latent_low', 'lm'], "
    msg += "current name: %s" % config.model_name
    raise Exception(msg)

  model = Model(config)
  with tf.variable_scope(config.model_name):
    model.build()

  # controller
  controller = Controller(config)
  if(config.model_name != "lm"): 
    if("lm" in controller.eval_metrics_list): controller.build_lm(LM, config)  
  controller.train(model, dset)
  return 

if __name__ == "__main__":
  main()
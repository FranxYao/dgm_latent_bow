import os

import tensorflow as tf 
import numpy as np 

from config import Config
from controller import Controller
from data_utils import Dataset
from seq2seq import Seq2seq

import time

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("controller_mode", None,
  "training task, 'train' or 'test'")

flags.DEFINE_integer("start_epoch", -1,
  "The index of the start epoch")

flags.DEFINE_integer("num_epoch", -1,
  "Number of training epoch")

flags.DEFINE_integer("batch_size", -1,
  "Size of the batch")

flags.DEFINE_integer("train_print_interval", -1,
  "The print interval during training")

flags.DEFINE_string("optimizer", "", "The optimizer")
flags.DEFINE_float("learning_rate", -1., "The learning rate")

flags.DEFINE_string("model_name", "", "The name of the model")
flags.DEFINE_string("model_version", "", "The version of the model")
flags.DEFINE_string("gpu_id", "", "gpu index")

FLAGS.gpu_id = "0"
FLAGS.model_version = "test"

config = Config()
config.parse_arg(FLAGS)
config.print_arg()

# dataset
dset = Dataset(config)
dset.build()
config.vocab_size = dset.vocab_size
config.dec_start_id = dset.word2id["_GOO"]
config.dec_end_id = dset.word2id["_EOS"]

# model 
Model = Seq2seq
model = Model(config)
model.build()

# training 
start_epoch = config.start_epoch
num_epoch = config.num_epoch
batch_size = config.batch_size
model_name = config.model_name
drop_out = config.drop_out
print_interval = config.train_print_interval

os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
sess = tf.Session(config=gpu_config)
sess.run(tf.global_variables_initializer())

num_batches = dset.num_batches(batch_size, "train")
print("%d batches in total" % num_batches)
ei = 0
start_time = time.time()
loss = []

for bi in range(10):
  batch_dict = dset.next_batch("train", batch_size, model_name)
  batch_dict["drop_out"] = drop_out
  output_dict = model.train_step(sess, batch_dict)

  loss.append(output_dict["loss"])

  if(bi % print_interval == 0):
    print("e-%d, b-%d, t-%.2f, l-%.4f" % 
      (ei, bi, time.time() - start_time, output_dict["loss"]))

print("\nepoch %d, time %.2f, average loss %.4f\n" % 
  (ei, time.time() - start_time, np.average(loss)))

# predict 
batch_dict = dset.next_batch("dev", batch_size, "eval")
batch_dict["drop_out"] = 0.
output_dict = model.predict_greedy(sess, batch_dict)
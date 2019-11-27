"""The configuration"""

import os 
import shutil

class Config:
  ## Data configuration 
  dataset = "quora" # ["mscoco", "quora", 'wikibio']

  # For a detailed data processing clarification and consideration, see the 
  # data_utils.py
  dataset_path = {
    "mscoco": {
      "train": "../data/mscoco/captions_train2017.json",  
      "test": "../data/mscoco/captions_val2017.json"}, 
    'mscoco14': {
      'train': '../data/mscoco/annotations2014/captions_train2014.json',
      'test': '../data/mscoco/annotations2014/captions_train2014.json'
    },
    "quora": {
      "train": "../data/quora/train.txt", 
      "test": ""
    }
  }

  full_quora = False

  max_sent_len = {"mscoco": 16, 
                  'mscoco14': 16, 
                  "quora": 20} # 95 percentile 
  max_enc_len = {'wikibio': 85}
  max_dec_len = {'wikibio': 42}
  vocab_size = -1

  dec_start_id = 0
  dec_end_id = 1
  pad_id = 2
  unk_id = 3
  stop_words = None

  # the wikibio dataset configuration 
  data2text_dir = '../wiki2bio/processed_data'
  data2text_limits = 0
  data2text_word_vocab_path = '../wiki2bio/original_data/word_vocab.txt'
  data2text_field_vocab_path = '../wiki2bio/original_data/field_vocab.txt'

  ## Model configuration 
  """Model names in: 
  ["seq2seq", "bow_seq2seq", "latent_bow", "vae", "hierarchical_vae", "lm"]
  """
  model_name = "latent_bow" 
  model_mode = "train" # ["train", "test"]
  model_version = "0.1"
  model_path = "../models/"
  output_path = "../outputs/"

  state_size = 500
  drop_out = 0.6

  # encoder
  bow_loss_fn = "nll" # "nll", "l1"
  bow_pred_method = "seq_tag" # "seq_tag", "mix_softmax", "seq2seq"
  num_paraphrase = 4 # 1 for quora, 4 for mscoco
  enc_layers = 2
  lambda_enc_loss = 1.0
  max_enc_bow = 30 # The number of bag of words, 25 for mscoco, 11 for quora
  no_residual = False

  # vae setting 
  vae_seq2seq = True
  lambda_kl = 0.1 
  prior = "gaussian" # "vmf" or "gaussian"

  # decoder 
  decoding_mode = "greedy"
  dec_layers = 2
  is_attn = True
  source_attn = True
  max_dec_bow = 30 # 10 for mscoco, 11 for quora 
  source_sample_ratio = 0.
  sample_size = 30 # 12 for mscoco, 11 for quora, 30 for wikibio
  sampling_method = "greedy" # "topk", "greedy"
  topk_sampling_size = 1
  predict_source_bow = True

  is_gumbel = False
  gumbel_samples = 3
  gumbel_tau = 0.5
  is_cheat = False
  copy = False
  num_pointers = 3
  bow_cond = False
  bow_cond_gate = False

  ## Controller configuration
  # system setting 
  gpu_id = "0"
  controller_mode = "train"
  save_ckpt = False
  lm_load_path = "/home/francis/hdd/Columbia/dgm_yf2470/models/lm_0.1.0/model-e7.ckpt"

  # training hyperparameters
  batch_size = 100 # 60 for the seq2seq model, effective batch size = 100 
  start_epoch = 0
  num_epoch = 20
  train_print_interval = 500

  # evaluation metrics
  # eval_metrics_list = ["bleu", "rouge", "ppl", "dist", "self_bleu", "jaccard"]
  # eval_metrics_list = ["bleu", "rouge", "mem_cover"]
  eval_metrics_list = ["bleu", "rouge"]
  log_metrics = ["predict_average_confident", "target_average"]
  write_output = True
  single_ref = False
  compare_outputs = True 

  # optimizer 
  learning_rate_decay = False
  random_seed = 15213
  target_metrics = "bleu_2" # ["ppl", "bleu_2"]
  optimizer = "Adam" 
  learning_rate = 0.0008
  learning_rate_enc = 0.001 # or 1e-3, sensitive 
  learning_rate_dec = 0.001

  def parse_arg(self, flags):
    """Parsing the commandline arguments, overwrite the default"""
    self.is_attn = flags.is_attn
    self.is_gumbel = flags.is_gumbel
    self.vae_seq2seq = flags.vae_seq2seq
    self.save_ckpt = flags.save_ckpt
    self.is_cheat = flags.is_cheat
    if(self.vae_seq2seq): self.is_attn = False
    self.single_ref = flags.single_ref
    self.no_residual = flags.no_residual
    self.copy = flags.copy
    self.bow_cond = flags.bow_cond
    self.bow_cond_gate = flags.bow_cond_gate
    if(flags.num_pointers != -1): self.num_pointers = flags.num_pointers
    if(flags.enc_layers != -1): self.enc_layers = flags.enc_layers
    self.dec_layers = self.enc_layers

    if(flags.dataset != ""): self.dataset = flags.dataset
    if(self.dataset == "quora"):
      self.num_paraphrase = 1
      self.max_enc_bow = 11
      self.max_dec_bow = 11
      self.sample_size = 11

    if(flags.state_size != -1): self.state_size = flags.state_size
    if(flags.prior != ""): self.prior = config.prior
    if(flags.lambda_kl != -1): self.lambda_kl = flags.lambda_kl
    if(flags.controller_mode != None): 
      self.controller_mode = flags.controller_mode
    if(flags.optimizer != ""): self.optimizer = flags.optimizer
    if(flags.bow_loss != ""): self.bow_loss = flags.bow_loss
    if(flags.topk_sampling_size != -1): 
      self.topk_sampling_size = flags.topk_sampling_size
    if(flags.drop_out != -1): self.drop_out = flags.drop_out
    if(flags.learning_rate != -1.): self.learning_rate = flags.learning_rate
    if(flags.learning_rate_enc != -1.): self.learning_rate_enc = flags.learning_rate_enc
    if(flags.learning_rate_dec != -1.): self.learning_rate_dec = flags.learning_rate_dec
    if(flags.start_epoch != -1): self.start_epoch = flags.start_epoch
    if(flags.num_epoch != -1): self.num_epoch = flags.num_epoch
    if(flags.batch_size != -1): self.batch_size = flags.batch_size
    if(flags.train_print_interval != -1): 
      self.train_print_interval = flags.train_print_interval
    if(flags.model_name != ""): self.model_name = flags.model_name
    if(flags.model_version != ""): self.model_version = flags.model_version
    if(flags.gpu_id != ""): self.gpu_id = flags.gpu_id
    return 

  def setup_path(self):
    model = self.model_name + "_" + self.model_version
    output_path = self.output_path + model 
    model_path = self.model_path + model

    if(os.path.exists(model_path)):
      inp = input(
        "model %s already existed, overwite[o]; continue[c] or exit[e]?\n" % model)

      if(inp == "o"):
        shutil.rmtree(model_path)
        os.mkdir(model_path)
        shutil.rmtree(output_path)
        os.mkdir(output_path)
      elif(inp == "e"):
        print("exiting the program, please rename the model")
        sys.exit(1)
      else: pass 
    else:
      os.mkdir(model_path)
      os.mkdir(output_path)
    
    self.model_path = model_path
    self.output_path = output_path + "/"
    return

  def print_arg(self):
    print("--------------------- Configuration ----------------------")
    print('dataset config:')
    print("  dataset: %s" % self.dataset)

    print('model config:')
    print("  model_name: %s" % self.model_name)
    print("  model_mode: %s" % self.model_mode)
    print("  model_path: %s" % self.model_path)
    print("  output_path: %s" % self.output_path)
    print("  model_version: %s" % self.model_version)
    print("  num_paraphrase: %d" % self.num_paraphrase)
    print("  lambda_enc_loss: %.2f" % self.lambda_enc_loss)
    print("  enc_layers: %d" % self.enc_layers)
    print("  dec_layers: %d" % self.dec_layers)
    print("  state_size: %d" % self.state_size)
    print("  is_attn: %s" % str(self.is_attn))
    print("  source_attn: %s" % str(self.source_attn))
    print("  drop_out: %.2f" % self.drop_out)
    print("  vae_seq2seq: %s" % str(self.vae_seq2seq))
    print("  lambda_kl: %.5f" % self.lambda_kl)
    print("  prior: %s" % self.prior)
    print("  sampling_method: %s" % self.sampling_method)
    print("  topk sampling size: %d" % self.topk_sampling_size)
    print("  bow_pred_method: %s" % self.bow_pred_method)
    print("  is_gumbel: %s" % str(self.is_gumbel))
    print("  gumbel_tau: %.3f" % self.gumbel_tau)
    print("  max_enc_bow: %d" % self.max_enc_bow)
    print("  max_dec_bow: %d" % self.max_dec_bow)
    print("  is_cheat: %s" % (self.is_cheat))
    print("  sample_size: %d" % self.sample_size)
    print("  source_sample_ratio: %.2f" % self.source_sample_ratio)
    print("  bow_loss_fn: %s" % self.bow_loss_fn)
    print('  no_residual: %s' % self.no_residual)
    print('  copy: %s' % self.copy)
    print('  num_pointers: %d' % self.num_pointers)
    print('  bow_cond: %s' % self.bow_cond)
    print('  bow_cond_gate: %s' % self.bow_cond_gate)

    print('controller config:')
    print("  controller_mode: %s" % self.controller_mode)
    print("  batch_size: %d" % self.batch_size)
    print("  start_epoch: %d" % self.start_epoch)
    print("  num_epoch: %d" % self.num_epoch)
    print("  train_print_interval: %d" % self.train_print_interval)
    print("  optimizer: %s" % self.optimizer)
    print("  learning_rate_decay: %s" % float(self.learning_rate_decay))
    print("  learning_rate: %.6g" % self.learning_rate)
    print("  learning_rate_enc: %.6g" % self.learning_rate_enc)
    print("  learning_rate_dec: %.6g" % self.learning_rate_dec)
    print('  single_ref: %s' % self.single_ref)
    print('  compare_outputs: %s' % self.compare_outputs)
    print("  gpu_id: %s" % self.gpu_id)
    print("----------------------------------------------------------")
    return 
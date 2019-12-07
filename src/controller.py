"""Training process controller

Yao Fu, Columbia University 
yao.fu@columabi.edu
Mar 05TH 2019
"""

import os
import time
import tensorflow as tf 
import numpy as np 
import tqdm
import rouge

from nltk.translate.bleu_score import corpus_bleu
from nltk.util import ngrams


def _build_lm_batch_dict(output_dict, start_id, end_id):
  """Decoder output to lm input"""
  dec_targets = output_dict["dec_predict"]
  dec_inputs = np.zeros_like(dec_targets)
  dec_inputs = np.transpose(dec_inputs, [1, 0]) # [B, T] -> [T, B]
  dec_inputs[0] = start_id
  dec_inputs[1:] = np.transpose(dec_targets, [1, 0])[: -1]
  dec_inputs = np.transpose(dec_inputs, [1, 0])

  dec_lens = []
  for di in dec_targets:
    is_eos = False
    for i in range(len(di)):
      if(di[i] == end_id):
        dec_lens.append(i)
        is_eos = True
        break
    if(is_eos == False): dec_lens.append(i)

  batch_dict = {"dec_inputs": dec_inputs, "targets": dec_targets, 
    "out_lens": np.array(dec_lens)}
  return batch_dict

def _cut_eos(predict_batch, eos_id):
  """cut the eos in predict sentences"""
  pred = []
  for s in predict_batch:
    s_ = []
    for w in s:
      if(w == eos_id): break
      s_.append(w)
    pred.append(s_)
  return pred

def _id_list_to_sent(id_list, id2word):
  s = []
  for i in id_list: s.append(id2word[i])
  return s 

class TrainingLog(object):
  def __init__(self, config):
    self.model = config.model_name
    self.output_path = config.output_path
    self.bow_pred_method = config.bow_pred_method
    self.vae_seq2seq = config.vae_seq2seq
    self.is_cheat = config.is_cheat
    if(self.model == "bow_seq2seq"):
      if(self.bow_pred_method == "seq2seq"):
        self.log = {
                    "loss": [0],
                    "enc_bow_loss": [0],
                    "enc_seq2seq_loss": [0],
                    "enc_loss": [0],  
                    "dec_loss": [0],
                    "pred_overlap_topk": [0],
                    "pred_overlap_confident": [0] ,
                    "target_support": [0],
                    "pred_confident_support": [0],
                    "target_average": [0],
                    "predict_average_confident": [0],
                    "precision_confident": [0],
                    "recall_topk": [0]}
      elif(self.bow_pred_method == "seq_tag"):
        if(self.is_cheat):
          self.log = {"dec_loss": [0]}
        else:
          self.log = {
                      "loss": [0],
                      "enc_bow_loss": [0],
                      "enc_loss": [0],  
                      "dec_loss": [0],
                      "pred_overlap_topk": [0],
                      "pred_overlap_confident": [0] ,
                      "target_support": [0],
                      "pred_topk_support": [0], 
                      "pred_confident_support": [0],
                      "target_average": [0],
                      "predict_average_confident": [0],
                      "precision_confident": [0],
                      "recall_confident": [0],
                      "precision_topk": [0], 
                      "recall_topk": [0]}

    elif(self.model == "latent_bow"):
      self.log = {
                  "loss": [],
                  "enc_loss": [],  
                  "dec_loss": [],

                  "pred_overlap_topk": [],
                  "pred_overlap_confident": [] ,

                  "pred_topk_support": [], 
                  "pred_confident_support": [],
                  "target_support": [],
                  
                  "predict_average_confident": [],
                  "target_average": [],
                  'pointer_ent': [],
                  'avg_max_ptr': [],
                  'avg_num_copy': [],
                  
                  "precision_confident": [],
                  "recall_confident": [0],
                  "precision_topk": [], 
                  "recall_topk": []}

    elif(self.model == "seq2seq"):
      if(self.vae_seq2seq):
        self.log = {"loss": [], "dec_loss": [], "kl_loss": []}
        # self.log = {"loss": []}
      else:
        self.log = {"loss": []}

    elif(self.model == "lm"):
      self.log = {"loss": [], "ppl": []}

  def update(self, output_dict):
    """Update the log"""
    for l in self.log: 
      if(l in output_dict): self.log[l].append(output_dict[l])
    return

  def print(self):
    """Print out the log"""
    s = ""
    for l in self.log: s += "%s: %.4f, " % (l, np.average(self.log[l]))
    print(s)
    print("")
    return 

  def write(self, ei, log_metrics=None):
    """Write the log for current epoch"""
    log_path = self.output_path + "epoch_%d.log" % ei
    print("Writing epoch log to %s ... " % log_path)
    with open(log_path, "w") as fd:
      log_len = len(self.log[list(self.log.keys())[0]])
      for i in range(log_len):
        for m in self.log:
          if(log_metrics == None): 
            fd.write("%s: %.4f\t" % (m, self.log[m][i]))
          else:
            if(m in log_metrics): fd.write("%s: %.4f\t" % (m, self.log[m][i]))
        fd.write("\n")
    return 

  def reset(self):
    """Reset the log"""
    for l in self.log: 
      if(self.bow_pred_method == "seq2seq"):
        self.log[l] = []
      else:
        self.log[l] = []
    return

class Controller(object):
  """The training, validation and evaluation controller"""

  def __init__(self, config):
    """Initialization from the configuration"""
    self.mode = config.controller_mode
    self.model_name = config.model_name
    self.model_name_version = config.model_name + "_" + config.model_version
    self.start_epoch = config.start_epoch
    self.num_epoch = config.num_epoch
    self.write_output = config.write_output
    self.batch_size = config.batch_size
    self.print_interval = config.train_print_interval
    self.gpu_id = config.gpu_id
    self.drop_out = config.drop_out
    self.dec_start_id = config.dec_start_id
    self.dec_end_id = config.dec_end_id
    self.model_path = config.model_path
    self.output_path = config.output_path
    self.random_seed = config.random_seed
    self.bow_pred_method = config.bow_pred_method
    self.train_log = TrainingLog(config)
    self.id2word = None
    self.target_metrics = config.target_metrics
    self.lm_load_path = config.lm_load_path
    self.rouge_evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'], max_n=2)
    self.save_ckpt = config.save_ckpt
    self.eval_metrics_list = config.eval_metrics_list
    self.log_metrics = config.log_metrics
    self.gumbel_samples = config.gumbel_samples
    self.is_gumbel = config.is_gumbel
    return 

  def build_lm(self, Model, config):
    """Build the language model for evaluation"""
    self.lm = Model(config)
    with tf.variable_scope("lm"):
      self.lm.build()
    return 

  def train(self, model, dset):
    """Train the model with the controller"""     
    print("Start training ...")

    os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_id
    tf.set_random_seed(self.random_seed)

    self.id2word = dset.id2word

    start_epoch = self.start_epoch
    num_epoch = self.num_epoch
    batch_size = self.batch_size
    model_name = self.model_name
    drop_out = self.drop_out
    print_interval = self.print_interval
    train_log = self.train_log
    target_metrics = self.target_metrics
    model_name_version = self.model_name_version

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    sess = tf.Session(config=gpu_config)
    sess.run(tf.global_variables_initializer())

    # restore lm for evaluation 
    if(model_name != "lm"): 
      if("lm" in self.eval_metrics_list):
        print("Restore the language model ... ")
        self.lm.model_saver.restore(sess, self.lm_load_path)

    # training preparation 
    num_batches = dset.num_batches(batch_size, "train")
    best_target_metrics = -100000
    best_epoch = -1
    print("%d batches in total" % num_batches)
    print("metrics of first 200 batchs are not reliable ")

    for ei in range(start_epoch, start_epoch + num_epoch):
      start_time = time.time()

      for bi in range(num_batches):
      # for bi in range(100):
        batch_dict = dset.next_batch("train", batch_size, model_name)
        batch_dict["drop_out"] = drop_out
        output_dict = model.train_step(sess, batch_dict, ei)
        train_log.update(output_dict)

        if(bi % 20 == 0): 
          print(".", end = " ", flush=True)
        if(bi % print_interval == 0):
          print("\n%s: e-%d, b-%d, t-%.2f" % 
            (model_name_version, ei, bi, time.time() - start_time))
          train_log.print()

      print("\n\nepoch %d training finished" % ei)

      if(ei >= 0):
        if(self.bow_pred_method == "seq2seq"):
          self.encoder_eval(model, dset, sess, "test")
        else: 
          metrics_dict = self.eval(model_name, model, dset, sess, "test", ei=ei)
          if(metrics_dict[target_metrics] > best_target_metrics):
            best_epoch = ei
            print("increase validation %s from %.4f to %.4f, update model" %
              (target_metrics, best_target_metrics, metrics_dict[target_metrics]))
            save_path = self.model_path + "/model-e%d.ckpt" % ei
            if(self.save_ckpt): 
              model.model_saver.save(sess, save_path)
              print("saving model to %s" % save_path)
            best_target_metrics = metrics_dict[target_metrics]
          else: 
            print("no performance increase, keep the best model at epoch %d" %
              best_epoch)
            print("best %s: %.4f" % (target_metrics, best_target_metrics))

      print("\nepoch %d, time cost %.2f\n" % (ei, time.time() - start_time))
      train_log.print()
      if(self.write_output): train_log.write(ei, self.log_metrics)
      train_log.reset()
      print("")
    return 

  def eval_metrics(self, sess, output_dict, batch_dict):
    """"""
    metrics_dict = {}
    if("bleu" in self.eval_metrics_list):
      metrics_dict.update(
        {"bleu_1": -1, "bleu_2": -1, "bleu_3": -1, "bleu_4": -1})
    if("ppl" in self.eval_metrics_list): 
      metrics_dict.update({"ppl": -1})
    if("jaccard" in self.eval_metrics_list):
      metrics_dict.update({"jaccard_dist": -1})
    if("self_bleu" in self.eval_metrics_list):
      metrics_dict.update(
        {"bleu_input_1": -1, "bleu_input_2": -1,
         "bleu_input_3": -1, "bleu_input_4": -1})
    if("rouge" in self.eval_metrics_list):
      metrics_dict.update({"rouge_1": -1, "rouge_2": -1, "rouge_l": -1})
    if("dist" in self.eval_metrics_list):
      metrics_dict.update({"dist_1": set(), "dist_2": set(), "dist_3": set()})
                    
    # perplexity, a measure of naturalness/ quality 
    if("ppl" in self.eval_metrics_list): 
      lm_batch_dict = _build_lm_batch_dict(
        output_dict, self.dec_start_id, self.dec_end_id)
      lm_eval_metrics = self.lm.eval_step(sess, lm_batch_dict)
      metrics_dict["ppl"] = lm_eval_metrics["ppl"]

    # generated BLEU v.s. reference, a measure of quality 
    predicts = _cut_eos(output_dict["dec_predict"], self.dec_end_id)
    reference = batch_dict["references"]

    if("bleu" in self.eval_metrics_list):
      metrics_dict["bleu_1"] = corpus_bleu(
        reference, predicts, weights=(1, 0, 0, 0))
      metrics_dict["bleu_2"] = corpus_bleu(
        reference, predicts, weights=(0.5, 0.5, 0, 0))
      metrics_dict["bleu_3"] = corpus_bleu(
        reference, predicts, weights=(0.33, 0.33, 0.34, 0))
      metrics_dict["bleu_4"] = corpus_bleu(
        reference, predicts, weights=(0.25, 0.25, 0.25, 0.25))

    # distinct ngrams, return as a set
    if("dist" in self.eval_metrics_list):
      unigrams = []
      bigrams = []
      trigrams = []
      for s in predicts: 
        unigrams.extend(list(ngrams(s, 1)))
        bigrams.extend(list(ngrams(s, 2)))
        trigrams.extend(list(ngrams(s, 3)))
      metrics_dict["dist_1"] = set([u[0] for u in unigrams])
      metrics_dict["dist_2"] = set([tuple(g) for g in bigrams])
      metrics_dict["dist_3"] = set([tuple(g) for g in trigrams])

    # rouge score
    if("rouge" in self.eval_metrics_list):
      rouge_pred = [" ".join(_id_list_to_sent(s, self.id2word)) for s in predicts]
      rouge_ref = []
      for r in reference:
        r_ = []
        for ri in r: r_.append(" ".join(_id_list_to_sent(ri, self.id2word)))
        rouge_ref.append(r_)
      rouge_scores = self.rouge_evaluator.get_scores(rouge_pred, rouge_ref)
      metrics_dict["rouge_1"] = rouge_scores["rouge-1"]["f"]
      metrics_dict["rouge_2"] = rouge_scores["rouge-2"]["f"]
      metrics_dict["rouge_l"] = rouge_scores["rouge-l"]["f"]

    # Jaccard Distance, lexical level diversity 
    if("jaccard" in self.eval_metrics_list):
      origin = batch_dict["enc_inputs"].T[1:].T
      origin = _cut_eos(origin, self.dec_end_id)
      jd = []
      for pred, orig in zip(predicts, origin):
        pred = set(pred)
        orig = set(orig)
        jd_ = float(len(pred & orig)) / len(pred | orig)
        jd.append(jd_)
      metrics_dict["jaccard_dist"] = np.average(jd)

    # generated BLEU v.s. origin, ngram diversity 
    if("self_bleu" in self.eval_metrics_list):
      reference = [[s] for s in origin]
      metrics_dict["bleu_input_1"] = corpus_bleu(
        reference, predicts, weights=(1, 0, 0, 0))
      metrics_dict["bleu_input_2"] = corpus_bleu(
        reference, predicts, weights=(0.5, 0.5, 0, 0))
      metrics_dict["bleu_input_3"] = corpus_bleu(
        reference, predicts, weights=(0.33, 0.33, 0.34, 0))
      metrics_dict["bleu_input_4"] = corpus_bleu(
        reference, predicts, weights=(0.25, 0.25, 0.25, 0.25))

    # matching score given by a matching model, semantic similarity 

    return metrics_dict

  def encoder_eval(self, model, dset, sess, mode):
    """Only evaluate the encoder for the bow_seq2seq_2seq model"""
    print("Evaluating the encoder ... ")

    start_time = time.time()
    batch_size = self.batch_size
    model_name = self.model_name

    num_batches = dset.num_batches(batch_size, mode)
    print("%d batches in total" % num_batches)

    metrics_dict = {"enc_infer_overlap": [],
                    "enc_infer_pred_support": [],
                    "enc_infer_target_support": [],
                    "enc_infer_precision": [],
                    "enc_infer_recall": []}
    pred_batch = np.random.randint(0, num_batches)

    for bi in range(num_batches):
      batch_dict = dset.next_batch(mode, batch_size, model_name)
      output_dict = model.enc_infer_step(sess, batch_dict)

      for m in output_dict: 
        if(m in metrics_dict): metrics_dict[m].append(output_dict[m])

    dset.print_predict_seq2paraphrase(output_dict, batch_dict)
    
    for m in metrics_dict: 
      metrics_dict[m] = np.average(metrics_dict[m])
      print("%s: %.4f" % (m , metrics_dict[m]))
    print("time cost: %.2fs" % (time.time() - start_time))
    print("")
    return metrics_dict

  def eval_generate(self, model, dset, sess, mode, decoding_mode="greedy", ei=-1):
    """Validation or evaluation"""
    print("Evaluating ... ")

    start_time = time.time()
    batch_size = self.batch_size
    model_name = self.model_name

    num_batches = dset.num_batches(batch_size, mode)
    print("%d batches in total" % num_batches)

    metrics_dict = {}
    if("bleu" in self.eval_metrics_list):
      metrics_dict.update(
        {"bleu_1": [], "bleu_2": [], "bleu_3": [], "bleu_4": []})
    if("ppl" in self.eval_metrics_list): 
      metrics_dict.update({"ppl": []})
    if("jaccard" in self.eval_metrics_list):
      metrics_dict.update({"jaccard_dist": []})
    if("self_bleu" in self.eval_metrics_list):
      metrics_dict.update(
        {"bleu_input_1": [], "bleu_input_2": [],
         "bleu_input_3": [], "bleu_input_4": []})
    if("rouge" in self.eval_metrics_list):
      metrics_dict.update({"rouge_1": [], "rouge_2": [], "rouge_l": []})
    if("dist" in self.eval_metrics_list):
      metrics_dict.update({"dist_1": set(), "dist_2": set(), "dist_3": set()})
    if("mem_cover" in self.eval_metrics_list):
      metrics_dict.update({"words_from_mem": [], "dec_output_bow_cnt": [], 
        "dec_mem_ratio": []})

    if(self.write_output):
      output_fd = open(self.output_path + "output_e%d.txt" % ei, "w")
    # pred_batch = np.random.randint(0, num_batches)
    pred_batch = 0

    references = []
    dec_outputs = []

    for bi in range(num_batches):
    # for bi in range(50):
      batch_dict = dset.next_batch(mode, batch_size, model_name)
      references.extend(batch_dict['references'])
      output_dict = model.predict(sess, batch_dict)    
      dec_outputs.extend(
        _cut_eos(output_dict["dec_predict"], self.dec_end_id))    

      if(self.write_output and 
        self.model_name == "latent_bow" and self.is_gumbel):
        output_dict_list = [output_dict]
        for i in range(self.gumbel_samples - 1):
          output_dict_list.append(model.predict(sess, batch_dict))
        dset.print_gumbel(output_dict_list, batch_dict, output_fd)
      else: 
        dset.print_predict(model_name, output_dict, batch_dict, 
          output_fd, 0)


      if(bi == pred_batch):
        print("")
        if(self.is_gumbel):
          output_dict_list = [output_dict]
          for i in range(self.gumbel_samples - 1):
            output_dict_list.append(model.predict(sess, batch_dict))
          dset.print_gumbel(output_dict_list, batch_dict)
        else:
          dset.print_predict(model_name, output_dict, batch_dict, None, 3)

      metrics_dict_update = self.eval_metrics(sess, output_dict, batch_dict)
      metrics_dict_update.update(output_dict)
      for m in metrics_dict_update: 
        if(m in metrics_dict): 
          if(m[:4] != "dist"): metrics_dict[m].append(metrics_dict_update[m])
          else: 
            if("dist" in self.eval_metrics_list): 
              metrics_dict[m] |= metrics_dict_update[m]
      if(bi % 20 == 0): print(".", end = " ", flush=True)
    print("")
    if(self.write_output): output_fd.close()

    # print('corpus level bleu:')
    # print(references[0])
    # print(dec_outputs[0])
    # print('bleu 1: %.4f' % 
    #   corpus_bleu(references, dec_outputs, weights=[1, 0, 0, 0]))
    # print('bleu 2: %.4f' % 
    #   corpus_bleu(references, dec_outputs, weights=[1, 1, 0, 0]))
    # print('bleu 3: %.4f' % 
    #   corpus_bleu(references, dec_outputs, weights=[1, 1, 1, 0]))
    # print('bleu 4: %.4f' % 
    #   corpus_bleu(references, dec_outputs, weights=[1, 1, 1, 1]))
    
    print('sentence level:')
    for m in metrics_dict :  
      if(m[:4] != "dist"): metrics_dict[m] = np.average(metrics_dict[m])
      else: 
        if("dist" in self.eval_metrics_list): 
          metrics_dict[m] = len(metrics_dict[m])
      print("%s: %.4f" % (m , metrics_dict[m]))

    print("time cost: %.2fs" % (time.time() - start_time))
    print("")
    return metrics_dict

  def eval_lm(self, model, dset, sess, mode, decoding_mode="greedy"):
    print("Evaluating the language model ... ")

    start_time = time.time()
    batch_size = self.batch_size
    model_name = self.model_name
    num_batches = dset.num_batches(batch_size, mode)
    print("%d batches in total" % num_batches)
    metrics_dict = {"loss": [], "ppl": []}

    for bi in range(num_batches):
      batch_dict = dset.next_batch(mode, batch_size, model_name)
      output_dict = model.eval_step(sess, batch_dict)
      for m in output_dict: 
        if(m in metrics_dict): metrics_dict[m].append(-output_dict[m])
    
    for m in metrics_dict: 
      metrics_dict[m] = np.average(metrics_dict[m])
      print("%s: %.4f" % (m , metrics_dict[m]))
    print("time cost: %.2fs" % (time.time() - start_time))
    print("")
    return metrics_dict

  def eval(self, 
    model_name, model, dset, sess, mode, decoding_mode="greedy", ei=-1):
    if(model_name == "lm"):
      metrics_dict = self.eval_lm(model, dset, sess, mode, decoding_mode)
    elif(model_name in ["seq2seq", "latent_bow", "vae", "bow_seq2seq"]):
      metrics_dict = self.eval_generate(
        model, dset, sess, mode, decoding_mode, ei)
    return metrics_dict

  def generate(self, model, sess, input_sent):
    """Generate the paraphrase controlling the latent variables"""
    return 
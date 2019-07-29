from data_utils import * 
from config import Config

# train_path = "../data/mscoco/annotations/captions_train2017.json"
# train_sentences = mscoco_read_json(train_path)
# corpus_statistics(train_sentences)
# train_sentences, dev_sentences = train_dev_split(train_sentences) 

# word2id, id2word = get_vocab(train_sentences)

# train_sentences, train_lens = normalize(train_sentences, word2id, 16)
# dev_sentences = normalize(dev_sentences, word2id, 15)

# test_path = "../data/mscoco/annotations/captions_val2017.json"
# test_sentences = mscoco_read_json(test_path)
# test_sentences = normalize(test_sentences, word2id, 15)

config = Config()
dset = Dataset(config)
dset.build()

batch_size = 20
print(dset.num_batches(batch_size, "train"))
print(dset.num_batches(batch_size, "dev"))
print(dset.num_batches(batch_size, "test"))

# batch_dict = dset.next_batch("train", batch_size, "seq2seq")

## Upper bound, use the target sentence 

## 

batch_dict = dset.next_batch("train", batch_size, "bow_seq2seq")

def test_batch_bow(batch_dict, id2word):
  i = 0
  for enc_inp, enc_tgt, dec_inp, dec_tgt in zip(batch_dict["enc_inputs"],
                                                batch_dict["enc_targets"], batch_dict["dec_bow"], 
                                                batch_dict["dec_targets"]):
    print("enc sentence: " + " ".join([id2word[w] for w in enc_inp]))
    print("enc targets: " + " ".join([id2word[w] for w in enc_tgt]))
    print("dec inputs: " + " ".join([id2word[w] for w in dec_inp]))
    print("dec targets: " + " ".join([id2word[w] for w in dec_tgt]))
    print("")
    i += 1
    if(i == 5): break

test_batch_bow(batch_dict, dset.id2word)

dset.decode_sent(batch_dict["enc_inputs"][0])
dset.decode_sent(batch_dict["references"][0][0])
dset.decode_sent(batch_dict["references"][0][1])
dset.decode_sent(batch_dict["references"][0][2])
dset.decode_sent(batch_dict["references"][0][3])

decode_sent(dset, batch_dict["references"][0][0])
decode_sent(dset, batch_dict["references"][0][1])

batch_dict = dset.next_batch("train", batch_size, "seq2seq")
dset.decode_sent(batch_dict["enc_inputs"][0])
dset.decode_sent(batch_dict["dec_inputs"][0])
dset.decode_sent(batch_dict["targets"][0])
# for _ in range(dset.num_batches(batch_size, "train")): batch_dict = dset.next_batch("test", batch_size, "seq2seq")
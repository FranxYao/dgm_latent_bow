import re, time, os

class Vocab(object):
    """vocabulary for words and field types"""
    def __init__(self, word_vocab_path, field_vocab_path):
        vocab = dict()
        vocab['_GOO'] = 0
        vocab['_EOS'] = 1
        vocab['_PAD'] = 2
        vocab['_UNK'] = 3
        cnt = 4
        with open(word_vocab_path, "r") as v:
        # with open("original_data/word_vocab.txt", "r") as v:
            for line in v:
                word = line.strip().split()[0]
                vocab[word] = cnt
                cnt += 1
        self.word2id = vocab
        self.id2word = {value: key for key, value in vocab.items()}

        key_map = dict()
        key_map['_GOO'] = 0
        key_map['_EOS'] = 1
        key_map['_PAD'] = 2
        key_map['_UNK'] = 3
        cnt = 4
        with open(field_vocab_path, "r") as v:
        # with open("original_data/field_vocab.txt", "r") as v:
            for line in v:
                key = line.strip().split()[0]
                key_map[key] = cnt
                cnt += 1
        self.key2id = key_map
        self.id2key = {value: key for key, value in key_map.items()}

    # def word2id(self, word):
    #     ans = self._word2id[word] if word in self._word2id else 3
    #     return ans

    # def id2word(self, id):
    #     ans = self._id2word[int(id)]
    #     return ans

    # def key2id(self, key):
    #     ans = self._key2id[key] if key in self._key2id else 3
    #     return ans

    # def id2key(self, id):
    #     ans = self._id2key[int(id)]
    #     return ans

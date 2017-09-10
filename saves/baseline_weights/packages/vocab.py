from collections import Counter
import pickle
import json

class Vocab(object):

    def __init__(self, dictionary_dir, max_oovs):
        
        with open(dictionary_dir,'r') as f:
            self.w2i = json.load(f)
        self.i2w = {v: k for k, v in self.w2i.items()}
        self.count = len(self.w2i)
        self.max_oovs = 12

    def word2idx(self, word):
        if word not in self.w2i:
            return self.w2i['<UNK>']
        return self.w2i[word]

    def idx2word(self, idx):
        if idx not in self.i2w:
            return '<UNK>'
        return self.i2w[idx]

    # changes an indices list to a word list
    # can include idx2oov dictionary to change oov words

    def create_oov_list(self, word_list):
        # creates oov2idx and idx2oov from a given sequence
        # max_oovs: maximum number of OOVs allowed
        oov2idx = {}
        idx2oov = {}
        oov_count=0
        for word in word_list:
            if (word not in oov2idx) & (word not in self.w2i):
                oov2idx[word] = self.count + oov_count
                idx2oov[self.count+oov_count] = word
                oov_count+=1
            if oov_count>=self.max_oovs:
                return oov2idx, idx2oov
        return oov2idx,idx2oov

    def tensor_to_string(self, tensor, oov2idx):
        idx_list = tensor.tolist()
        out_list = []
        for x in idx_list:
            if x==self.w2i['<EOS>']:
                break
            elif x==self.w2i['<PAD>']:
                continue
            elif x==self.w2i[';']:
                out_list.append(x)
                break
            else:
                out_list.append(x)
        # idx_list = [x for x in idx_list if x!=0]
        if len(out_list)==0:
            return 'None'
        idx2oov = {v: k for k, v in oov2idx.items()}
        word_list = self.idx_list_to_word_list(out_list,idx2oov)
        return ' '.join(word_list)
    
    def idx_list_to_word_list(self, idx_list, idx2oov={}):
        # idx2oov: contains {oov_idx: oov_word,  such as 50000:Thompson}
        if type(idx_list[0])!=int:
            idx_list = [int(idx) for idx in idx_list]
        out = []
        for idx in idx_list:
            if idx==3: # EOS
                break
            elif idx in idx2oov:
                out.append(idx2oov[idx])
            else:
                out.append(self.idx2word(idx))
        return out

    # changes a word list to an indices list
    # can preserve oov words by creating a temporary dictionary and index

    def word_list_to_idx_list(self, word_list, oov2idx={}):
        # oov2idx: temporary dictionary of the oov words introduced per sample
        out = []
        for word in word_list:
            if word in oov2idx:
                out.append(oov2idx[word])
            else:
                out.append(self.word2idx(word))
        return out

    def preprocess_string(self, text, preprocess_list):
        # when given a list of tuples (e.g. ("  ", " ")), preprocesses string
        for tup in preprocess_list:
            from_str, to_str = tup
            text = text.replace(from_str, to_str)
        return text

import torch
import os
from torch.utils import data
import numpy as np
from packages.functions import preprocess
import re

"""
Sample usage case
src_root = '/home/irteam/users/mjchoi/github/DL2DL/data/task1/source/'
trg_root = '/home/irteam/users/mjchoi/github/DL2DL/data/task1/target/'
data_loader = get_loader(src_root, trg_root, 2)
src,trg, src_len, trg_len = dat_iter.next()
"""


class TextFolder(data.Dataset):
    def __init__(self, root, dictionary, vocab, single):
        """
        Initializes paths and preprocessing module
        root: data directory
        dictionary: dictionary directory
        vocab: vocab object loaded
        single: whether our model trains from looking at a single sentence (default: False)
        """
        self.vocab = vocab

        with open(root) as f:
            self.data = f.read().split('\n')
        self.max_len = 100
        self.load_dict(dictionary)
        self.single = single
        self.splitter = re.compile(r"(\W)")

    def __getitem__(self, index):
        data = self.data[index].split('%NWL%')

        try:
            label = int(data[-1])  # label
        except ValueError:
            label = np.random.randint(0, max(1, len(data) - 2))

        if self.single:
            src_tokens = self.tokenize(data[:-2][label], 'single')  # single context
            src_tokens = src_tokens.split(' ')
        else:
            src_tokens = self.tokenize(data[:-2], 'multi')  # context

        target = [x for x in self.splitter.split(data[-2]) if x is not '']
        split_point = np.random.randint(1, min(len(target), 5))
        qry_ = target[:split_point]
        qry_tokens = self.tokenize(qry_, 'single')  # guery
        trg_ = target[split_point:]
        trg_tokens = self.tokenize(trg_, 'target')  # target

        if self.single:
            oov2idx, _ = self.vocab.create_oov_list(src_tokens + qry_tokens + trg_tokens)
            src_np = self.vocab.word_list_to_idx_list(src_tokens, oov2idx)
        else:
            # create an OOV list from a single src / qry / trg set
            oov2idx, _ = self.vocab.create_oov_list(self.flatten(src_tokens) + qry_tokens + trg_tokens)
            # lists of words to lists of idxs
            src_tokens = [self.vocab.word_list_to_idx_list(line, oov2idx) for line in src_tokens]
            max_src = max([len(x) for x in src_tokens])

            # source tokens to np arrays (to fix sequence length)
            src_np = np.zeros([len(src_tokens), max_src], dtype=int)
            for i, tokens in enumerate(src_tokens):
                src_np[i, :len(tokens)] = np.array(tokens, dtype=int)
            diff = 10 - src_np.shape[0]
            if diff > 0:
                src_np = np.vstack([src_np, np.zeros([diff, src_np.shape[1]], dtype=int)])
        # query and target tokens to num lists
        qry_tokens = self.vocab.word_list_to_idx_list(qry_tokens, oov2idx)
        trg_tokens = self.vocab.word_list_to_idx_list(trg_tokens, oov2idx)

        return torch.LongTensor(src_np), torch.LongTensor(qry_tokens), \
               torch.LongTensor(trg_tokens), label, oov2idx

    def __len__(self):
        return len(self.data)

    def flatten(self, listoflist):
        function = lambda l: [item for sublist in l for item in sublist]
        return function(listoflist)

    def load_dict(self, dictionary):
        # load dictionary
        import json
        with open(dictionary) as f:
            txt = f.read()
            self.w2i = json.loads(txt)
            self.i2w = {v: k for k, v in self.w2i.items()}

    def wordlist2idxlist(self, wordlist):  # idx lists created here should contain OOV info
        out = []
        for word in wordlist:
            if word in self.w2i:
                out.append(self.w2i[word])
            else:
                out.append(self.w2i['<UNK>'])
        return out

    def tokenize(self, input, mode=None):
        if mode == 'multi':  # input: list of strings
            out = [[x_ for x_ in self.splitter.split(x)[:self.max_len] if x_ is not ''] for x in input]
        elif mode == 'single':  # input: string
            out = input[:self.max_len]
        elif mode == 'target':  # input: string
            out = ['<SOS>'] + input[:self.max_len - 2] + ['<EOS>']
        return out


def collate_fn(data):
    # Sort function: sorts in decreasing order by the length of the items in the right (targets)
    data.sort(key=lambda x: len(x[1]), reverse=True)
    sources, queries, targets, labels, oovs = zip(*data)

    """
    -- Inputs --
    1. sources: list of batch @ [10 x seq] tensors OR list of batch @ [seq] tensors
    2. queries: list of batch @ [seq] tensors
    3. targets: list of batch @ [seq] tensors
    4. labels: list of bqtch @ [integers], which line to point in a sentence
    
    -- Outputs --
    1. context_len: a list of len batch, lengths of all contexts (mostly 10)
    2. sources_out: a tensor of size [batch*10-a x seq], all input sentences merged into 1 matrix
    3. source_len: a list of len batch*10-a, length of every line in 1
    4. queries_out: a tensor of size [batch x seq], all queries merged into 1 matrix
    5. query_len: a list of len batch, lengths of all queries
    6. targets_out: a tensor of size [batch x seq], all outputs merged into 1 matrix
    7. target_len: a list of len batch, lenghts of all answers
    """
    if sources[0].dim() == 1:  # if only given 1 line to look at
        source_len = [x.size(0) for x in sources]
        sources_out = torch.zeros(len(sources), max(source_len)).long()
        for i, source in enumerate(sources):
            sources_out[i, :len(source)] = source
        context_len = None
    elif sources[0].dim() == 2:  # select best answer from N(=10) lines
        context_len = [x.size(0) for x in sources]  # number of sentences for each sample
        source_len = []
        for x in sources:
            for line in x:
                source_len.append(len(line))
        # source_len = [x.size(1) for x in sources] # max size of sequence length
        sources_out = []
        t_l = 0
        max_ = max(source_len)
        for source in sources:
            added = max_ - source.size(1)
            if added > 0:
                sources_out.append(torch.cat([source, torch.zeros(source.size(0), added).long()], 1))
            else:
                sources_out.append(source)
        sources_out = torch.cat(sources_out, 0)
    # get required lengths
    query_len = [len(x) for x in queries]
    target_len = [len(x) for x in targets]

    queries_out = torch.zeros(len(queries), max(query_len)).long()
    targets_out = torch.zeros(len(targets), max(target_len)).long()
    for i in range(len(queries_out)):
        queries_out[i, :len(queries[i])] = queries[i]
        targets_out[i, :len(targets[i])] = targets[i]

    # source_len = np.maximum(np.array(source_len),150).tolist()
    # query_len = np.maximum(np.array(query_len),150).tolist()
    # sources_out = sources_out[:,:150]
    # queries_out = queries_out[:,:150]
    outputs = (sources_out, queries_out, targets_out)
    lengths = (source_len, query_len, target_len, context_len)

    return outputs, lengths, labels, list(oovs)


def get_loader(root, dictionary, vocab, batch_size=64, single=False, num_workers=2, shuffle=True):
    dataset = TextFolder(root, dictionary, vocab, single)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return data_loader

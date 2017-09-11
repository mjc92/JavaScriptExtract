import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from packages.functions import to_cuda
from torch.autograd import Variable
from models.copynet import CopyNet
from models.baselines.bigru import biGRU_attn
import time

# encoder model for copynet

class JavascriptExtractor(nn.Module):
    def __init__(self, args, vocab):
        super(JavascriptExtractor, self).__init__()
        self.args = args
        if args.model=='copynet':
            self.copynet = CopyNet(args, vocab)
        elif args.model=='bigru':
            self.bigru = biGRU_attn(len(vocab.w2i),args.embed,args.hidden,1,2,args.cuda)
        if args.single==False:
            # set similarity measurement
            if args.encoder=='position':
                from models.similarity.position_encoding import PositionEncoding
                self.encoder = PositionEncoding(args, vocab)
            elif args.encoder=='lstm':
                from models.similarity.lstm import LSTMLastState
                self.encoder = LSTMLastState(args, vocab)

            if args.similarity=='cosine':
                from models.similarity.cosine import CosineSimilarity
                self.similarity = CosineSimilarity(args, vocab)
            elif args.similarity=='mlp':
                from models.similarity.mlp import MLPSimilarity
                self.similarity = MLPSimilarity(args, vocab)
            elif args.similarity=='levenshtein':
                from models.similarity.levenshtein import LevenshteinDistance
                self.similarity = LevenshteinDistance(args)
            elif args.similarity=='position_cosine':
                from models.similarity.position_cosine import PositionCosine
                self.position_cosine = PositionCosine(args, vocab)
            elif args.similarity=='lstm_cosine':
                from models.similarity.lstm_cosine import LSTMCosine
                self.lstm_cosine = LSTMCosine(args, vocab)

        self.iscuda = args.cuda
        self.single = args.single
        self.vocab = vocab
        self.d_hid = args.hidden
        self.d_emb = args.embed

    def forward(self, sources, queries, lengths, targets):
        """
        sources: [batch*context_lines x seq] OR [batch x seq]
        queries: [batch x seq]
        targets: [batxh x seq]
        """
        source_len,query_len,target_len,context_len = lengths

        # use similarity function to get closest lines from source
        if self.single==False:
            # similarity_encode
            # similarity_compute
            if self.args.similarity=='levenshtein':
                sources, similarities = self.similarity(sources, queries, context_len)
            elif self.args.similarity=='position_cosine':
                sources, similarities = self.position_cosine(sources, queries, context_len)
            elif self.args.similarity=='lstm_cosine':
                sources, similarities = self.lstm_cosine(sources, queries, context_len)
            else:
                src_simil, q_simil = self.encoder(sources, queries)
                sources, similarities = self.similarity(sources, src_simil, q_simil, context_len)
                # here, 'sources' are the selected lines

        # merge sources and queries to one matrix
        source_lens = (sources>0).long().sum(1)
        query_lens = (queries>0).long().sum(1)
        max_len = (source_lens+query_lens).max()
        new_sources = torch.zeros(sources.size(0),max_len).long()
        new_sources = to_cuda(new_sources, self.iscuda)
        for i in range(sources.size(0)):
            try:
                new_sources[i,:source_lens[i]] += sources[i,:source_lens[i]]
            except ValueError:
                pass
            new_sources[i,source_lens[i]:(source_lens[i]+query_lens[i])] += queries[i,:query_lens[i]]


        # get target outputs using the copynet model
        if self.args.model=='copynet':
            outputs = self.copynet(new_sources, targets)
        elif self.args.model=='bigru':
            outputs = self.bigru(new_sources, targets)

        # delete irrelevant items
        del sources, targets, source_len, query_len, target_len, context_len, lengths
        del new_sources

        if self.single:
            return outputs
        else:
            return outputs, similarities

class JavascriptSimilarity(nn.Module):
    # only gets similarity
    def __init__(self, args, vocab):
        super(JavascriptSimilarity, self).__init__()
        self.args = args

        if args.single==False:
            # set similarity measurement
            if args.encoder=='position':
                from models.similarity.position_encoding import PositionEncoding
                self.encoder = PositionEncoding(args, vocab)
            elif args.encoder=='lstm':
                from models.similarity.lstm import LSTMLastState
                self.encoder = LSTMLastState(args, vocab)

            if args.similarity=='cosine':
                from models.similarity.cosine import CosineSimilarity
                self.similarity = CosineSimilarity(args, vocab)
            elif args.similarity=='mlp':
                from models.similarity.mlp import MLPSimilarity
                self.similarity = MLPSimilarity(args, vocab)
            elif args.similarity=='levenshtein':
                from models.similarity.levenshtein import LevenshteinDistance
                self.similarity = LevenshteinDistance(args)

        self.iscuda = args.cuda
        self.single = args.single
        self.vocab = vocab
        self.d_hid = args.hidden
        self.d_emb = args.embed

    def forward(self, sources, queries, lengths, targets):
        """
        sources: [batch*context_lines x seq] OR [batch x seq]
        queries: [batch x seq]
        targets: [batxh x seq]
        """
        source_len,query_len,target_len,context_len = lengths

        # use similarity function to get closest lines from source
        src_simil, q_simil = self.encoder(sources, queries)
        sources, similarities = self.similarity(sources, src_simil, q_simil, context_len)
        return similarities

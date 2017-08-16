import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from packages.functions import to_cuda
from torch.autograd import Variable
from models.copynet import CopyNet
import time

# encoder model for copynet

class JavascriptExtractor(nn.Module):
    def __init__(self, args, vocab):
        super(JavascriptExtractor, self).__init__()
           
        self.copynet = CopyNet(args, vocab)

        # set similarity measurement
        if args.similarity=='position':
            from models.similarity.position_encoding import PositionEncoding
            self.similarity = PositionEncoding(args, vocab)
        
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
            sources, similarities = self.similarity(sources, queries, context_len)
        # merge sources and queries to one matrix
        source_lens = (sources>0).long().sum(1)
        query_lens = (queries>0).long().sum(1)
        max_len = (source_lens+query_lens).max()
        new_sources = torch.zeros(sources.size(0),max_len).long()
        new_sources = to_cuda(new_sources, self.iscuda)
        for i in range(sources.size(0)):
            new_sources[i,:source_lens[i]] += sources[i,:source_lens[i]]
            new_sources[i,source_lens[i]:source_lens[i]+query_lens[i]] += queries[i,:query_lens[i]]
        
        self.inputs = new_sources # inputs for the copynet model
        # get target outputs using the copynet model
        outputs = self.copynet(new_sources, targets)
        
        return outputs, similarities
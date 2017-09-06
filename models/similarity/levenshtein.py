import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from packages.functions import to_cuda

class LevenshteinDistance(nn.Module):
    # compares directly using Levenshtein distance
    def __init__(self, args):
        super(LevenshteinDistance, self).__init__()
        self.iscuda = args.cuda

    def forward(self, sources, queries, context_len):
        
        
        bc, in_seq = sources.size()
        b = len(context_len)
        
        idx = 0
        similarity_list = []
        sources_list = []
        for i,c in enumerate(context_len):
            query = queries[i].tolist()
            similarities = [self.lev(line.tolist(),query) for line in sources[idx:idx+c]]
            similarities = [x+1e-4 for x in similarities]
            similarities = torch.Tensor(similarities)
            
            similarity_list.append(similarities) # distribution of each line, for later softmax
            max_idx = similarities.max(0)[1]
            sources_list.append(sources[idx+max_idx.data[0]]) # selected source
            idx+=c
        
        sources = torch.stack(sources_list,0) # [b x seq]
        
        similarity_tensor = to_cuda(Variable(torch.zeros(b,10)),self.iscuda)
        for i,sim in enumerate(similarity_list):
            length = len(sim)
            similarity_tensor[i,:length] = similarity_tensor[i,:length] + sim
        
        return sources, similarity_tensor
        bc, in_seq = sources.size()
        b = len(context_len)
    
    def lev(self, a, b):
        if not a: return len(b)
        if not b: return len(a)
        return min(self.lev(a[1:], b[1:])+(a[0] != b[0]), self.lev(a[1:], b)+1, self.lev(a, b[1:])+1)
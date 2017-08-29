import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from packages.functions import to_cuda

class CosineSimilarity(nn.Module):
    # relies only on positional encodings
    def __init__(self, args, vocab):
        super(CosineSimilarity, self).__init__()
        self.iscuda = args.cuda

    def forward(self, sources, src_simil, q_simil, context_len):
        
        bc, in_seq = sources.size()
        b = len(context_len)
        
        idx = 0
        similarity_list = []
        sources_list = []
        for i,c in enumerate(context_len):
            similarities = F.softmax(torch.mm(
                src_simil[idx:idx+c],q_simil[i].unsqueeze(1)).squeeze())
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
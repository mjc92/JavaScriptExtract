import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from packages.functions import to_cuda

class MLPSimilarity(nn.Module):
    # relies only on positional encodings
    def __init__(self, args, vocab):
        super(MLPSimilarity, self).__init__()
        self.embed = args.embed
        self.iscuda = args.cuda
        
        self.mlp = nn.Sequential(
            nn.Linear(self.embed*2,1),
            nn.ReLU())

    def forward(self, sources, src_simil, q_simil, context_len):
        bc, in_seq = sources.size()
        b = len(context_len)
        
        idx = 0
        similarity_list = []
        sources_list = []
        for i,c in enumerate(context_len):
            similarities = torch.cat([src_simil[idx:idx+c],
                  q_simil[i].unsqueeze(0).expand(c,q_simil.size(1))],1)
            similarities = self.mlp(similarities).squeeze()
            similarities = F.softmax(similarities)
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
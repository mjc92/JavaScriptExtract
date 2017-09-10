import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from packages.functions import to_cuda

class LSTMCosine(nn.Module):
    # encodes all states of an LSTM
    def __init__(self, args, vocab):
        super(LSTMCosine, self).__init__()
        self.embedding = nn.Embedding(vocab.count,args.embed)
        self.embed = args.embed
        self.max_in_seq = args.max_in_seq
        self.vocab = vocab
        self.iscuda = args.cuda
        self.lstm = nn.LSTM(input_size = self.embed, hidden_size = self.embed,
                            batch_first = True)
        self.cos = nn.modules.CosineSimilarity(0)

    def forward(self, sources, queries, context_len):
        """
        sources: LongTensor, [batch*context x src_seq]
        queries: LongTensor, [batch x qry_seq]
        context_len: LongTensor, [batch]
        """
        bc, in_seq = sources.size()
        b, q_seq = queries.size()

        embedded_sources = self.embedding(to_cuda(Variable(self.unk_tensor(sources)),self.iscuda))
        embedded_queries = self.embedding(to_cuda(Variable(self.unk_tensor(queries)),self.iscuda))

        encoded_sources, _ = self.lstm(embedded_sources)
        encoded_queries, _ = self.lstm(embedded_queries)

        src_mask = Variable((sources>0).float().unsqueeze(2)) # [batch*context x src_seq x 1]
        q_mask = Variable((queries>0).float().unsqueeze(2)) # [batch x qry_seq x 1]
        q_len = q_mask.squeeze().sum(1).data.long().tolist() # [batch]
        src_mask = to_cuda(src_mask,self.iscuda)
        q_mask = to_cuda(q_mask,self.iscuda)

        c_idx = 0
        source_list = []
        sim_list = []
        # print(context_len)
        # print(q_len)
        for i in range(b):
            # print(i,context_len[i],c_idx)
            tmp1 = encoded_sources[c_idx:c_idx+context_len[i],:q_len[i]]
            tmp2 = encoded_queries[i,:q_len[i]]
            sim = F.softmax((tmp1*tmp2).sum(2).sum(1))
            sim_list.append(sim)
            top_score = sim.max(0)[1].data[0] # argmax
            source_list.append(sources[c_idx+top_score]) # add answer
            c_idx += context_len[i]

        # get similarities
        similarities = torch.stack(sim_list,0)
        sources = torch.stack(source_list,0)

        return sources, similarities

    def unk_tensor(self, tensor):
        unk = self.vocab.w2i['<UNK>']
        mask = (tensor>=self.vocab.count).long()
        ones = torch.ones(mask.size()).long()
        ones = to_cuda(ones, self.iscuda)
        tensor = tensor * (ones-mask) + mask * unk
        return tensor

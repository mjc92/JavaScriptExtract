import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from packages.functions import to_cuda

class LSTMLastState(nn.Module):
    # relies on the last state of an LSTM
    def __init__(self, args, vocab):
        super(LSTMLastState, self).__init__()
        self.embedding = nn.Embedding(vocab.count,args.embed)
        self.embed = args.embed
        self.max_in_seq = args.max_in_seq
        self.pos_emb = self.get_pos_embedding() # [max_seq_len x embed_dim]
        self.vocab = vocab
        self.iscuda = args.cuda
        self.lstm = nn.LSTM(input_size = self.embed, hidden_size = self.embed,
                            batch_first = True)

    def forward(self, sources, queries):
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
        
#         # here we will use the last hidden state
        source_len = (sources>0).long().sum(1)
        query_len = (queries>0).long().sum(1)
        
        sources_last = [x[source_len[i]-1] for i,x in enumerate(encoded_sources)]
        queries_last = [x[query_len[i]-1] for i,x in enumerate(encoded_queries)]
        
        src_simil = torch.stack(sources_last,0)
        q_simil = torch.stack(queries_last,0)
        return src_simil, q_simil
        
        
        
#         y_list = []
#         for i,length in enumerate(context_len):
#             y_list.append(queries_last[i].expand(length,hidden))
#         x = torch.cat(sources_last,0)
#         y = torch.cat(y_list,0) # [batch*context x hidden]
#         mul = F.cosine_similarity(x,y) # [batch*context]
        
#         temp = 0
#         idx_list = []
#         attn_list = []
#         source_list = []
#         encoded_list = []
#         for i,length in enumerate(context_len):
#             attn = F.softmax(mul[temp:temp+length])
#             attn_list.append(attn)
#             idx = attn.max(0)[1].data[0]
#             idx_list.append(idx)
#             out = (encoded_sources[temp:temp+length] * attn.unsqueeze(1).unsqueeze(2)).sum(0)
#             source_list.append(sources[temp+idx].unsqueeze(0))
#             encoded_list.append(out.unsqueeze(0))
#             temp += length
#         out = torch.cat(encoded_list,0)
#         attns = torch.cat(attn_list,0)
#         sources = torch.stack(source_list,0)
#         return out, sources, attns, idx_list

    def get_pos_embedding(self):
        out = torch.zeros(self.max_in_seq, self.embed)
        for j in range(self.max_in_seq):
            for k in range(self.embed):
                out[j,k] = (1-j/self.max_in_seq) - (k/self.embed) * (1 - 2*j/self.max_in_seq)
        return out
        
    def unk_tensor(self, tensor):
        unk = self.vocab.w2i['<UNK>']
        mask = (tensor>=self.vocab.count).long()
        ones = torch.ones(mask.size()).long()
        ones = to_cuda(ones, self.iscuda)
        tensor = tensor * (ones-mask) + mask * unk
        return tensor

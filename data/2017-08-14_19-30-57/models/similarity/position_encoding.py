import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from packages.functions import to_cuda

class PositionEncoding(nn.Module):
    # relies only on positional encodings
    def __init__(self, args, vocab):
        super(PositionEncoding, self).__init__()
        self.embedding = nn.Embedding(vocab.count,args.embed)
        self.embed = args.embed
        self.max_in_seq = args.max_in_seq
        self.pos_emb = self.get_pos_embedding() # [max_seq_len x embed_dim]
        self.vocab = vocab
        self.iscuda = args.cuda

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
        
        src_mask = Variable((sources>0).float().unsqueeze(2))
        q_mask = Variable((queries>0).float().unsqueeze(2))
        src_mask = to_cuda(src_mask,self.iscuda)
        q_mask = to_cuda(q_mask,self.iscuda)
        
        sources_out = embedded_sources * to_cuda(
            Variable(self.pos_emb[:in_seq]).unsqueeze(0).expand(bc,in_seq,self.embed),self.iscuda)
        queries_out = embedded_queries * to_cuda(
            Variable(self.pos_emb[:q_seq]).unsqueeze(0).expand(b,q_seq,self.embed),self.iscuda)
        
        # get resulting tensors of shape [bc x embed] & [b x embed]
        src_simil = (sources_out * src_mask).sum(1)
        q_simil = (queries_out * q_mask).sum(1)
        
        idx = 0
        similarity_list = []
        sources_list = []
        for i,c in enumerate(context_len):
            similarities = F.softmax(torch.mm(src_simil[idx:idx+c],q_simil[i].unsqueeze(1)).squeeze())
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
        
        
        # here we will use the last hidden state
        source_len = (sources>0).long().sum(1)
        sources_last = [x[source_len[i]-1].unsqueeze(0) for i,x in enumerate(encoded_sources)]
        queries_last = [x[query_len[i]-1].unsqueeze(0) for i,x in enumerate(encoded_queries)]
        
        
        
        y_list = []
        for i,length in enumerate(context_len):
            y_list.append(queries_last[i].expand(length,hidden))
        x = torch.cat(sources_last,0)
        y = torch.cat(y_list,0) # [batch*context x hidden]
        mul = F.cosine_similarity(x,y) # [batch*context]
        
        temp = 0
        idx_list = []
        attn_list = []
        source_list = []
        encoded_list = []
        for i,length in enumerate(context_len):
            attn = F.softmax(mul[temp:temp+length])
            attn_list.append(attn)
            idx = attn.max(0)[1].data[0]
            idx_list.append(idx)
            out = (encoded_sources[temp:temp+length] * attn.unsqueeze(1).unsqueeze(2)).sum(0)
            source_list.append(sources[temp+idx].unsqueeze(0))
            encoded_list.append(out.unsqueeze(0))
            temp += length
        out = torch.cat(encoded_list,0)
        attns = torch.cat(attn_list,0)
        sources = torch.stack(source_list,0)
        return out, sources, attns, idx_list

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

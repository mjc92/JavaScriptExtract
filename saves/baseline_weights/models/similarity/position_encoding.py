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

    def forward(self, sources, queries):
        """
        sources: LongTensor, [batch*context x src_seq]
        queries: LongTensor, [batch x qry_seq]
        """
        bc, in_seq = sources.size()
        b, q_seq = queries.size()

        embedded_sources = self.embedding(to_cuda(Variable(self.unk_tensor(sources)),self.iscuda))
        embedded_queries = self.embedding(to_cuda(Variable(self.unk_tensor(queries)),self.iscuda))

        src_mask = Variable((sources>0).float().unsqueeze(2))
        q_mask = Variable((queries>0).float().unsqueeze(2))
        src_mask = to_cuda(src_mask,self.iscuda)
        q_mask = to_cuda(q_mask,self.iscuda)

        in_seq = min([in_seq,q_seq]) # for truncated
        sources_out = embedded_sources[:,:in_seq] * to_cuda( # for truncated
        # sources_out = embedded_sources * to_cuda(
            Variable(self.pos_emb[:in_seq]).unsqueeze(0).expand(bc,in_seq,self.embed),self.iscuda)
        queries_out = embedded_queries * to_cuda(
            Variable(self.pos_emb[:q_seq]).unsqueeze(0).expand(b,q_seq,self.embed),self.iscuda)

        # get resulting tensors of shape [bc x embed] & [b x embed]
        # src_simil = (sources_out * src_mask).sum(1)
        src_simil = (sources_out * src_mask[:,:sources_out.size(1)]).sum(1) # for truncated
        q_simil = (queries_out * q_mask).sum(1)

        return src_simil, q_simil

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

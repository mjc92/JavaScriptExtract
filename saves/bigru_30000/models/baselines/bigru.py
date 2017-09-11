import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

# RNN Based Language Model
class biGRU_attn(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, sos, is_cuda):
        super(biGRU_attn, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.GRU(embed_size, hidden_size, num_layers,
                                batch_first=True, bidirectional=True)
        self.decoder = nn.GRU(embed_size + hidden_size*2, hidden_size, num_layers,
                                batch_first=True)
        self.W1 = nn.Linear(hidden_size*2, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size*2)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
        self.sos = sos
        self.hidden_size = hidden_size
        self.is_cuda = is_cuda
        self.vocab_size = vocab_size

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)

    def forward(self, enc_in, dec_in, teacher_forcing=True):

        # remove OOVs
        enc_mask = (enc_in>=self.vocab_size).long()
        dec_mask = (dec_in>=self.vocab_size).long()
        enc_in = (1-enc_mask)*enc_in + enc_mask
        dec_in = (1-dec_mask)*dec_in + dec_mask

        # to Variable
        enc_in = Variable(enc_in)
        dec_in = Variable(dec_in)
        if self.is_cuda:
            enc_in = enc_in.cuda()
            dec_in = dec_in.cuda()

        # Encoder
        enc_embedded = self.embed(enc_in)
        encoded, _ = self.encoder(enc_embedded)
        # Decoder
        dec_embedded = self.embed(dec_in)
        state = self.W1(encoded[:,-1]).unsqueeze(0)
        outputs = []
        context = Variable(torch.FloatTensor(dec_in.size(0),
                    1,self.hidden_size*2).zero_()) # get initial context, [b x 1 x h*2]
        if self.is_cuda:
            context = context.cuda()
        for i in range(dec_in.size(1)-1):
            if teacher_forcing==True:
                input = torch.cat([context,dec_embedded[:,i].unsqueeze(1)],dim=2)
            else:
                if i==0:
                    input = dec_embedded[:,0].unsqueeze(1)
                else:
                    next_words = self.linear(out.squeeze())
                    next_idx = next_words.max(1)[1]
                    input = self.embed(next_idx).unsqueeze(1)
                input = torch.cat([context,input],dim=2)
            out, state = self.decoder(input, state)
            comp = self.W2(state) # [batch x hidden*2]
            scores = torch.bmm(encoded,comp.view(comp.size(1),-1,1)) # [b x seq x 1]
            scores = F.softmax(scores)
            context = torch.bmm(scores.view(scores.size(0),1,-1),encoded) # [b x 1 x h*2]
            outputs.append(out)
        outputs = torch.cat(outputs,dim=1) # [b x seq x h]

        # outputs = outputs.contiguous().view(-1, outputs.size(2))
        # Decode hidden states of all time step
        outputs = self.linear(outputs)
        outputs = outputs.view(dec_in.size(0),-1,self.vocab_size)
        return outputs
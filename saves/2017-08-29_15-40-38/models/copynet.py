import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import Counter
from packages.functions import to_cuda
from models.decoder import CopyDecoder
# normal copynet model: takes in idx matrix, spits out targets

class CopyNet(nn.Module):
    def __init__(self, args, vocab):
        super(CopyNet, self).__init__()
        
        # if args.encoder=='lstm':
        #     self.encoder = nn.LSTM(input_size=args.embed, 
        #                            hidden_size=args.hidden, num_layers=1, batch_first=True,
        #                            bidirectional=True)
        # elif args.encoder=='gru':
        #     self.encoder = nn.GRU(input_size=args.embed, 
        #                           hidden_size=args.hidden, num_layers=1, batch_first=True)
        # elif args.encoder=='transformer':
        #     from models.transformer.Models import Encoder
        #     self.encoder = Encoder(
        #         n_max_seq=args.max_in_seq, n_layers=args.n_layers, n_head=args.n_head,
        #         d_word_vec=args.embed, d_model=args.hidden,
        #         d_inner_hid=args.hidden*2, dropout=0.1)
        self.transformer = True if args.encoder=='transformer' else False

        self.encoder = nn.GRU(input_size=args.embed, 
                              hidden_size=args.hidden, num_layers=1, batch_first=True,
                              bidirectional=True)
        
        self.embedding = nn.Embedding(vocab.count, args.embed)
        
        self.decoder = CopyDecoder(args, vocab, self.embedding)

        self.iscuda = args.cuda
        self.single = args.single
        self.vocab = vocab
        self.d_hid = args.hidden
        self.d_emb = args.embed

    def forward(self, sources, targets):
        """
        sources: LongTensor, context + queries
        targets: LongTensor, targets (w/ SOS, EOS)
        """
        ################################ Encoder ################################ 
        unked_sources = Variable(self.unk_tensor(sources))
        unked_targets = Variable(self.unk_tensor(targets))
        if self.iscuda:
            unked_sources = unked_sources.cuda()
            unked_targets = unked_targets.cuda()        
        embedded_sources = self.embedding(unked_sources)
        embedded_targets = self.embedding(unked_targets)
        
        if self.transformer==True:
            encoded_sources = self.encoder(embedded_sources, sources)
        else:
            encoded_sources,_ = self.encoder(embedded_sources)
        
        ################################ Decoder ################################ 
        outputs = self.decoder(encoded_sources, sources, targets)
        return outputs[:,:-1] # exclude last one
        
        
    def unk_tensor(self, tensor):
        unk = self.vocab.w2i['<UNK>']
        mask = (tensor>=self.vocab.count).long()
        ones = torch.ones(mask.size()).long()
        ones = to_cuda(ones, self.iscuda)
        tensor = tensor * (ones-mask) + mask * unk
        return tensor

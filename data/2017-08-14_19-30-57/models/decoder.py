import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from packages.functions import to_cuda
from collections import Counter

class CopyDecoder(nn.Module):
    def __init__(self, args, vocab, embedding):
        super(CopyDecoder, self).__init__()
        self.vocab = vocab
        self.vocab_size = vocab.count
        self.hidden = args.hidden
        self.iscuda = args.cuda
        self.gru = nn.GRU(input_size = args.embed + args.hidden*2, hidden_size = args.hidden,
                          batch_first=True)
        self.max_oovs = args.max_oovs
        self.max_out_seq = args.max_out_seq
        self.is_train = args.mode=='train'
        
        self.embedding = embedding # word embedding matrix
        self.Ws = nn.Linear(self.hidden*2, self.hidden) # for getting initial state
        self.Wc = nn.Linear(self.hidden*2, self.hidden)
        self.Wo = nn.Linear(self.hidden, self.vocab_size)
        
    def forward(self, encoded_sources, sources, targets=None):
        """
        embedding: embedding function from above
        encoded_sources: Variable, [batch x seq x hidden]
        sources, targets: LongTensor, [batch x seq]
        """
        vocab_size = self.vocab_size
        hidden_size = self.hidden
        b, seq, _ = encoded_sources.size()
        
        source_lens = (sources>0).long().sum(1)
        
        if targets is not None:
            self.max_out_seq = targets.size(1)
            target_lens = (targets>0).long().sum(1)
        
        # 0. set initial states
        last_step = torch.stack([x[source_lens[i]-1] for i,x in enumerate(encoded_sources)],0) # [batch x hidden*2]
        state = self.Ws(last_step).unsqueeze(0) # [1 x batch x hidden*2]
        weighted = Variable(torch.Tensor(b, 1 , hidden_size * 2).zero_()) # [b x 1 x hidden]
        weighted = to_cuda(weighted, self.iscuda)

        
        out_list = []
        for i in range(self.max_out_seq):
            # 1. update states
            if self.is_train:
                inputs = self.embedding(Variable(self.unk_tensor(targets[:,i])))
            gru_input = torch.cat([inputs.unsqueeze(1),weighted],2) # [b x 1 x h+h]
            _, state = self.gru(gru_input, state) # [ 1 x b x hidden]
            
            # 2. predict next word y_t
            # 2-1) get score_g
            score_g = self.Wo(state.squeeze()) # [b x vocab_size]
            
            # 2-2) get score_c
            score_c = F.tanh(self.Wc(encoded_sources.contiguous().view(-1,hidden_size * 2)))
            score_c = score_c.view(b,-1,hidden_size) # [b x seq x hid]
            score_c = torch.bmm(score_c, state.view(b,-1,1)).squeeze() # [b x seq]
            score_c = F.tanh(score_c)
            encoded_mask = Variable((sources==0).float()*(-1000)) # causing inplace error
            score_c = score_c + encoded_mask

            # 2-3) get softmax-ed probs
            score = torch.cat([score_g, score_c],1) # [b x (vocab+seq)]
            probs = F.softmax(score)
            prob_g = probs[:,:vocab_size]
            prob_c = probs[:,vocab_size:]
            
            ############################################################################################################

            # 2-4) add to prob_g slots for OOVs
            oovs = Variable(torch.Tensor(b,self.max_oovs).zero_())+1e-5
            oovs = to_cuda(oovs,self.iscuda)
            prob_g = torch.cat([prob_g, oovs],1)
            
            # 2-5) add prob_c to prob_g
            numbers = sources.view(-1).tolist()
            set_numbers = list(set(numbers)) # unique numbers that appear
            c = Counter(numbers)
            dup_list = [k for k in set_numbers if (c[k]>1)]
            dup_attn_sum = Variable(torch.zeros(b,seq))
            masked_idx_sum = Variable(torch.Tensor(b,seq).zero_())
            encoded_idx_var = Variable(sources)
            if self.iscuda:
                dup_attn_sum = dup_attn_sum.cuda()
                masked_idx_sum = masked_idx_sum.cuda()
                encoded_idx_var = encoded_idx_var.cuda()
            
            for dup in dup_list:
                mask = (encoded_idx_var==dup).float()
                masked_idx_sum += mask
                attn_mask = torch.mul(mask,prob_c)
                attn_sum = attn_mask.sum(1).unsqueeze(1)
                dup_attn_sum += torch.mul(mask, attn_sum)
                
            attn = torch.mul(prob_c,(1-masked_idx_sum))+dup_attn_sum
            batch_indices = torch.arange(start=0, end=b).long()
            batch_indices = batch_indices.expand(seq,b).transpose(1,0).contiguous().view(-1)
            idx_repeat = torch.arange(start=0, end=seq).repeat(b).long()
            prob_c_to_g = Variable(torch.zeros(b,self.vocab_size+self.max_oovs))
            word_indices = sources.view(-1)
            if self.iscuda:
            #     batch_indices = batch_indices.cuda()
            #     idx_repeat = idx_repeat.cuda()
            #     prob_c_to_g = prob_c_to_g.cuda()
                attn = attn.cpu()
                word_indices = word_indices.cpu()

            prob_c_to_g[batch_indices,word_indices] += attn[batch_indices,idx_repeat]
            if self.iscuda:
                prob_c_to_g = prob_c_to_g.cuda()
                attn = attn.cuda()
            # 2-6) get final output
            out = prob_g + prob_c_to_g
                        
            # 3. get weighted attention to use for predicting next word
            # 3-1) get tensor that shows whether each decoder input has previously appeared in the encoder
            
            prev_input = (targets[:,i]).unsqueeze(1).expand(b,sources.size(1))
            idx_from_input = (sources == prev_input).float()
            idx_from_input = Variable(idx_from_input)
            for j in range(b):
                if idx_from_input[j].sum().data[0]>1:
                    idx_from_input[j] = idx_from_input[j]/idx_from_input[j].sum().data[0]

            # 3-2) multiply with prob_c to get final weighted representation
            weight_attn = prob_c * idx_from_input
            weight_attn = weight_attn.unsqueeze(1) # [b x 1 x seq]
            weighted = torch.bmm(weight_attn, encoded_sources) # weighted: [b x 1 x hidden]
            
            # 4. get next inputs
            max_vals = self.unk_tensor(out.max(1)[1].data)
            inputs = self.embedding(Variable(max_vals))
            
            out_list.append(out) # out_seq @ [batch x vocab+oov]

        # get final outputs
        return torch.stack(out_list,1)
    
    def unk_tensor(self, tensor):
        unk = self.vocab.w2i['<UNK>']
        mask = (tensor>=self.vocab.count).long()
        ones = torch.ones(mask.size()).long()
        ones = to_cuda(ones,self.iscuda)
        tensor = tensor * (ones-mask) + mask * unk
        return tensor
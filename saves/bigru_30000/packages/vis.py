from packages.data_loader import get_loader
from torch.autograd import Variable
import torch
from packages.functions import pack_padded, write_log
import os
import numpy as np
import time

def vis(model, vocab, args):
    mode = 'Validation results:'
    data_loader = get_loader(args.val_root, args.dict_root, vocab, args.batch,
                             args.single, shuffle=False)

    for i, (inputs, lengths, labels, oovs) in enumerate(data_loader):
        model.eval()
        sources, queries, targets = inputs
        # np.savetxt('saves/vis/sources.csv',sources[0].numpy(),delimiter=',')
        np.savetxt('saves/vis/targets.csv',targets[0,1:].numpy(),delimiter=',')
        sources = sources
        sources_tok = vocab.idx_list_to_word_list(sources[0].tolist(),oovs[0])
        sources_tok = [x for x in sources_tok if x!='<PAD>']
        print("sources: %s" %' '.join(sources_tok))
        # with open('saves/vis/sources_tok.csv','w') as f:
        #     f.write(','.join(sources_tok))


        targets_tok = vocab.idx_list_to_word_list(targets[0,1:].tolist(),oovs[0])
        targets_tok = [x for x in targets_tok if x!='<PAD>']
        print("targets: %s" %' '.join(targets_tok))
        with open('saves/vis/targets_tok.csv','w') as f:
            f.write(','.join(targets_tok))
        source_len, query_len, target_len, context_len = lengths
        if args.cuda:
            sources = sources.cuda()
            queries = queries.cuda()
            targets = targets.cuda()
        if args.single:
            outputs = model(sources, queries, lengths, targets)  # [batch x seq x vocab]
        else:
            outputs, sim = model(sources, queries, lengths, targets)
        outputs = outputs[0].cpu().data.max(1)[1]
        outputs_tok = vocab.idx_list_to_word_list(outputs.tolist(),oovs[0])
        targets_tok = [x for x in targets_tok if x!='<PAD>']
        print("outputs: %s" %' '.join(outputs_tok))
        with open('saves/vis/outputs_tok.csv','w') as f:
            f.write(','.join(outputs_tok))
        np.savetxt('saves/vis/outputs.csv',outputs.numpy(),delimiter=',')
        print('delay...')
        time.sleep(3)
        print('delay ended!')
    return

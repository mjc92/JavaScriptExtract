from packages.data_loader import get_loader
from torch.autograd import Variable
import torch
from packages.functions import pack_padded, to_np, to_var, write_log
import os

def val_sim(model, vocab, args, steps=None):
    mode = 'Validation results:'
    data_loader = get_loader(args.val_root, args.dict_root, vocab, args.batch,
                             args.single, shuffle=False)
    total_labels = 0
    total_correct_labels = 0

    for i, (inputs, lengths, labels, oovs) in enumerate(data_loader):
        model.eval()
        sources, queries, targets = inputs
        source_len, query_len, target_len, context_len = lengths
        if args.cuda:
            sources = sources.cuda()
            queries = queries.cuda()
            targets = targets.cuda()
        sim = model(sources, queries, lengths, targets)

        if args.single==False:
            predicted_labels = sim.data.max(1)[1]
            if args.cuda:
                correct_labels = (predicted_labels == torch.LongTensor(list(labels)).cuda()).sum()
            else:
                correct_labels = (predicted_labels == torch.LongTensor(list(labels))).sum()
            total_correct_labels += correct_labels
            total_labels += len(predicted_labels)



    acc1 = (total_correct_labels * 1.0 / total_labels)
    if steps is not None:
        string = "[%d] accuracy - Similarity: %1.3f" % (steps, acc1)
    print(string)
    write_log(string, args,'log_val.txt')


    # print("top-%d accuracy: %1.3f" %(args.k, total_correct_k*1.0/total_cases))

    return

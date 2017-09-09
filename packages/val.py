from packages.data_loader import get_loader
from torch.autograd import Variable
import torch
from packages.functions import pack_padded, write_log
import os


def val(model, vocab, args):
    mode = 'Validation results:'
    data_loader = get_loader(args.val_root, args.dict_root, vocab, args.batch,
                             args.single, shuffle=False)
    total_cases = 0
    total_correct_cases = 0
    total_correct_k = 0
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
        if args.single:
            outputs = model(sources, queries, lengths, targets)  # [batch x seq x vocab]
        else:
            outputs, sim = model(sources, queries, lengths, targets)

        targets = Variable(targets[:, 1:])  # correct answers

        packed_outputs, packed_targets = pack_padded(outputs, targets)

        del outputs, targets

        packed_outputs = packed_outputs.data
        packed_targets = packed_targets.data
        predicted = packed_outputs.max(1)[1] # how much was predicted
        unks = torch.LongTensor(predicted.size()).zero_()+1
        if args.cuda:
            unks = unks.cuda()
        unk_same = ((predicted == packed_targets)&(predicted==unks)).long().sum()
        eos_same = ((predicted == packed_targets)&(predicted==(unks+2))).long().sum()
        correct = ((predicted == packed_targets)).long().sum()

        print('%d out of %d tokens correct'%(correct-unk_same-eos_same,len(packed_targets)-eos_same))
        total_correct_cases += correct-unk_same-eos_same
        total_cases += len(packed_targets)-eos_same


        if args.single==False:
            predicted_labels = sim.data.max(1)[1]
            if args.cuda:
                correct_labels = (predicted_labels == torch.LongTensor(list(labels)).cuda()).sum()
            else:
                correct_labels = (predicted_labels == torch.LongTensor(list(labels))).sum()
            print('%d out of %d line predictions correct'%(correct_labels,len(predicted_labels)))
            total_correct_labels += correct_labels
            total_labels += len(predicted_labels)


        # free memory
        del packed_targets, packed_outputs, correct
        if not args.single:
            del sim, labels, correct_labels

            # get top-k accuracy
            # target_list = packed_targets.data
            # topk = packed_outputs.data.topk(args.k)[1]
            #         for i in range(len(target_list)):
            #             if target_list[i] in topk[i]:
            #                 total_correct_k += 1

    print("Total: %d out of %d tokens correct"%(total_correct_cases,total_cases))

    if args.single:
        acc = (total_correct_cases * 1.0 / total_cases)
        string = "%s accuracy: %1.3f" % (mode, acc)
    else:
        acc1 = (total_correct_labels * 1.0 / total_labels)
        acc2 = (total_correct_cases * 1.0 / total_cases)
        string = "%s accuracy - Similarity: %1.3f\tSequence: %1.3f" % (mode, acc1, acc2)
    print(string)
    write_log(string, args,'log_val.txt')

    # print("top-%d accuracy: %1.3f" %(args.k, total_correct_k*1.0/total_cases))

    return

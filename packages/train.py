from packages.vocab import Vocab
from packages.val import val
from tensorboard.logger import Logger
import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.autograd import Variable
from packages.data_loader import get_loader
from models.extractor import JavascriptExtractor
from packages.functions import pack_padded, to_np, to_var, write_log
import os

def train(args):
    print(args)
    if args.log:
        if args.log_dir is not None:
            logger = Logger(args.log_dir)
        else:
            logger = Logger('./logs')

    # load vocabulary and data loader
    vocab = Vocab(args.dict_root, args.max_oovs)
    data_loader = get_loader(args.train_root, args.dict_root, vocab, args.batch, args.single)

    # get loss and optimizer
    criterion = nn.NLLLoss()
    if args.load is None:
        model = JavascriptExtractor(args, vocab)
    else:
        model = torch.load(args.load)
    if args.cuda:
        model.cuda()
    steps = 0
    opt = optim.Adam(model.parameters(), lr=args.lr)
    total_batches = 0

    # train model for N epochs
    for epoch in range(args.epochs):
        within_steps = 0
        # load data samples
        for i, (inputs, lengths, labels, oovs) in enumerate(data_loader):
            # split tuples
            steps += 1
            if steps == 100000:
                sys.exit()
            total_batches = max(total_batches, i)
            # reset gradients
            model.zero_grad()
            sources, queries, targets = inputs
            source_len, query_len, target_len, context_len = lengths

            if args.cuda:
                sources = sources.cuda()
                queries = queries.cuda()
                targets = targets.cuda()
            # run model to get outputs
            if args.single:
                outputs = model(sources, queries, lengths, targets)  # [batch x seq x vocab]
            else:
                outputs, sim = model(sources, queries, lengths, targets)  # [batch x seq x vocab]
            # wrap targets with a Variable
            targets = Variable(targets[:, 1:])
            packed_outputs, packed_targets = pack_padded(outputs, targets)
            packed_outputs = torch.log(packed_outputs)
            # get loss for single / multiple lines
            if args.single:
                loss = criterion(packed_outputs, packed_targets)
            else:
                sim = sim + 1e-3
                sim = torch.log(sim)
                labels = Variable(torch.LongTensor(list(labels)))
                if args.cuda:
                    labels = labels.cuda()
                loss1 = criterion(sim, labels)
                loss2 = criterion(packed_outputs, packed_targets)
                loss = loss1 + loss2

            # get predicted values to get training accuracy
            for num in range(10):
                idx2oov = {k:v for v,k in oovs[num].items()}
                print('target: %s'%' '.join(vocab.idx_list_to_word_list(targets[num].data.tolist(),idx2oov)))
                print('output: %s'%' '.join(vocab.idx_list_to_word_list(outputs[num].max(1)[1].data.tolist(),idx2oov)))


            predicted = packed_outputs.data.max(1)[1]
            unks = torch.LongTensor(predicted.size()).zero_()+1
            if args.cuda:
                unks = unks.cuda()
            unk_same = ((predicted == packed_targets.data)&(predicted==unks)).long().sum()
            eos_same = ((predicted == packed_targets.data)&(predicted==(unks+2))).long().sum()
            correct = ((predicted == packed_targets.data)).long().sum()
            print('correct:',correct)
            print('unks:',unk_same)
            print('eos:',eos_same)
            acc = ((correct-unk_same-eos_same) * 1.0 / (packed_targets.size(0)-unk_same-eos_same))
            if args.single:
                string = "[%d]: Epoch %d\t%d/%d\tSequence - loss: %1.3f, acc: %1.3f" \
                         % (steps, epoch + 1, i, total_batches, loss.data[0], acc)
                print(string)
                write_log(string, args,'log_train.txt')

            else:
                predicted_label = sim.max(1)[1]
                correct_label = (predicted_label.data == labels.data).long().sum()
                acc2 = (correct_label * 1.0 / len(labels))
                string = "[%d]: Epoch %d\t%d/%d\tSimilarity - loss: %1.3f, acc: %1.3f\tSequence - loss: %1.3f, acc: %1.3f" \
                         % (steps, epoch + 1, i, total_batches, loss1.data[0], acc2, loss2.data[0], acc)
                print(string)
                write_log(string, args,'log_train.txt')
            loss.backward()

            # free memory
            del targets, outputs, packed_targets, packed_outputs, correct
            if not args.single:
                del sim, labels, correct_label
            opt.step()

            # get intermediate test results
            if steps % 100 == 0:
                val(model,vocab,args)
                torch.save(obj=model, f=os.path.join(args.save_dir, 'model_%d_steps.pckl' % steps))
                print("Model saved...")

            if args.log and steps % 10 == 0:
                # log scalar values
                info = {'loss': loss.data[0],
                        'tr_acc': acc}
                for tag, value in info.items():
                    logger.scalar_summary(tag, value, steps)

                # log values and gradients of the parameters
#                for tag, value in model.named_parameters():
#                    tag = tag.replace('.', '/')
#                    logger.histo_summary(tag, to_np(value), steps)
#                    logger.histo_summary(tag + '/grad', to_np(value.grad), steps)

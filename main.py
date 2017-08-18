import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import datetime
import torch
from torch import optim, nn
import argparse
import os
from packages.vocab import Vocab
import torch.nn.functional as F
from tensorboard.logger import Logger
from torch.autograd import Variable
from packages.data_loader import get_loader
from models.extractor import JavascriptExtractor
from packages.functions import pack_padded, to_np, to_var, str2bool, write_log

parser = argparse.ArgumentParser()

# arguments related to the dataset
# parser.add_argument('--train_root',type=str,default='/home/irteam/users/data/D3/outputs_train.txt',help='data file')
# parser.add_argument('--val_root',type=str,default='/home/irteam/users/data/D3/outputs_val.txt',help='data file')
# parser.add_argument('--test_root',type=str,default='/home/irteam/users/data/D3/outputs_test.txt',help='data file')
parser.add_argument('--train_root',type=str,default='data/outputs_train.txt',help='data file')
parser.add_argument('--val_root',type=str,default='data/outputs_val.txt',help='data file')
parser.add_argument('--test_root',type=str,default='data/outputs_test.txt',help='data file')
parser.add_argument('--dict_root',type=str,default='data/dict_1000.json',
                    help='directory of dictionary file')
parser.add_argument('--save_dir',type=str,default='saves', help='where to save model & info')
parser.add_argument("--max_oovs",type=int, default=20,
                    help='max number of OOVs to accept in a sample')

# arguments related to model training and inference
parser.add_argument("--mode",type=str, help='train/test mode. Error if unspecified')
parser.add_argument("--epochs",type=int, default=10, help='Number of epochs. Set by default to 20')
parser.add_argument("--lr",type=float, default=0.01, help='learning rate')
parser.add_argument("--batch",type=int, default=64, help='batch size')
parser.add_argument("--k",type=int, default=5, help='for top-k accuracy')
parser.add_argument("--single",type=str2bool, default=True, 
                    help='whether to include the task of selecting from multiple lines')
parser.add_argument("--cuda",type=str2bool, default=True, help='whether to use cuda')
parser.add_argument("--log",type=str2bool, default=False, help='whether to use tensorboard')
parser.add_argument("--load",type=str, default=None, help='whether to load from a previous model')
parser.add_argument("--copy",type=str2bool, default=False, help='whether to copy all related files to a folder')

# arguments related to the model structure itself
parser.add_argument("--hidden",type=int, default=256, help='size of hidden dimension')
parser.add_argument("--embed",type=int, default=256, help='size of embedded word dimension')
parser.add_argument("--n_layers",type=int, default=2, help='number of layers for transformer model')
parser.add_argument("--n_head",type=int, default=8, help='number of heads for transformer model')
parser.add_argument("--max_in_seq",type=int, default=100, help='max length of input')
parser.add_argument("--max_out_seq",type=int, default=100, help='max length of output')
parser.add_argument("--similarity",type=str, default='cosine', help='similarity measure to use')
parser.add_argument("--encoder",type=str, default='lstm', help='encoder type to use')

args = parser.parse_args()


time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
args.time_str = time_str

def train(args):
    print(args)
    if args.log:
        logger = Logger('./logs')
    vocab = Vocab(args.dict_root, args.max_oovs)
    data_loader = get_loader(args.train_root, args.dict_root, vocab, args.batch, args.single)

    criterion = nn.NLLLoss()
    if args.load is None:
        model = JavascriptExtractor(args,vocab)
    else:
        model = torch.load(args.load)
    if args.cuda:
        model.cuda()
    steps = 0
    opt = optim.Adam(model.parameters(), lr=args.lr)
    total_batches=0
    for epoch in range(args.epochs):
        within_steps = 0
        for i, (inputs, lengths, labels, oovs) in enumerate(data_loader):
            # split tuples
            steps+=1
            if steps==100000:
                sys.exit()
            total_batches = max(total_batches,i)
            model.zero_grad()
            sources, queries, targets = inputs
            source_len, query_len, target_len, context_len= lengths
            if args.cuda:
                sources = sources.cuda()
                queries = queries.cuda()
                targets = targets.cuda()
            if args.single:
                outputs = model(sources,queries,lengths, targets) # [batch x seq x vocab]
            else:
                outputs, sim = model(sources,queries,lengths, targets) # [batch x seq x vocab]
            targets = Variable(targets[:,1:])
            packed_outputs,packed_targets = pack_padded(outputs,targets)
            packed_outputs = torch.log(packed_outputs)
            if args.single:
                loss = criterion(packed_outputs,packed_targets)
            else:
                sim = sim + 1e-3
                sim = torch.log(sim)
                labels = Variable(torch.LongTensor(list(labels)))
                if args.cuda:
                    labels = labels.cuda()
                loss1 = criterion(sim, labels)
                loss2 = criterion(packed_outputs,packed_targets)
                loss = loss1 + loss2
                # loss = loss1
            predicted = packed_outputs.data.max(1)[1]
            correct=(predicted==packed_targets.data).long().sum()
            acc = (correct*1.0/packed_targets.size(0))
            if args.single:
                string = "[%d]: Epoch %d\t%d/%d\tLoss: %1.3f\tAccuracy: %1.3f"\
                      %(steps,epoch+1,i,total_batches,loss.data[0],acc)
                print(string)
                write_log(string, args)
                
            else:
                predicted_label = sim.max(1)[1]
                correct_label=(predicted_label.data==labels.data).long().sum()
                acc2 = (correct_label*1.0/len(labels))
                string = "[%d]: Epoch %d\t%d/%d\tLoss: %1.3f, %1.3f\tAccuracy: %1.3f, %1.3f"\
                      %(steps,epoch+1,i,total_batches,loss1.data[0],loss2.data[0],acc2,acc)
                print(string)
                write_log(string,args)
            loss.backward()
            
            # free memory
            del targets, outputs, packed_targets, packed_outputs, correct
            if args.single==False:
                del sim, labels, correct_label
            opt.step()
            if steps%100==0:
                # val(model,vocab, args)
                torch.save(obj=model,f=os.path.join(args.save_dir,'model_%d_steps.pckl'%steps))
                # print("Model saved...")
                if args.log:
                    # log scalar values
                    info = {'loss': loss.data[0],
                            'acc': acc}
                    for tag,value in info.items():
                        logger.scalar_summary(tag,value,steps)

                    # log values and gradients of the parameters
                    for tag, value in model.named_parameters():
                        tag = tag.replace('.','/')
                        logger.histo_summary(tag, to_np(value), steps)
                        logger.histo_summary(tag+'/grad',to_np(value.grad), steps)
                        
def val(model, vocab, args):
    criterion = nn.NLLLoss()
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
        source_len, query_len, target_len, context_len= lengths
        if args.cuda:
            sources = sources.cuda()
            queries = queries.cuda()
            targets = targets.cuda()
        if args.single:
            outputs = model(sources,queries,lengths, targets) # [batch x seq x vocab]
        else:
            outputs, sim = model(sources,queries,lengths,targets)
            
        targets = Variable(targets[:,1:]) # correct answers

        packed_outputs,packed_targets = pack_padded(outputs,targets)
        packed_outputs = torch.log(packed_outputs)
        if args.single:
            loss = criterion(packed_outputs,packed_targets).data[0]
        else:
            sim = sim + 1e-3
            sim = torch.log(sim)
            labels = Variable(torch.LongTensor(list(labels)))
            if args.cuda:
                labels = labels.cuda()
            loss1 = criterion(sim, labels).data[0]
            loss2 = criterion(packed_outputs,packed_targets).data[0]
            loss = loss1 + loss2
            # loss = loss1
        
        predicted = packed_outputs.max(1)[1]
        correct=(predicted==packed_targets).long().sum().data[0]
        
        total_correct_cases += correct
        total_cases += len(packed_outputs)
        
        print("Loss: %1.3f"%(loss))
        if args.single==False:
            predicted_label = sim.max(1)[1]
            correct_label=(predicted_label==labels).long().sum().data[0]
            
            total_correct_labels += correct_label
            total_labels += len(predicted_label)
            
        # free memory
        del targets, outputs, packed_targets, packed_outputs, correct
        if args.single==False:
            del sim, labels, correct_label

        # get top-k accuracy
        target_list = packed_targets.data
        topk = packed_outputs.data.topk(args.k)[1]
        for i in range(len(target_list)):
            if target_list[i] in topk[i]:
                total_correct_k += 1
    
    if args.single:
        acc = (total_correct_cases*1.0/total_cases)
        print("%s accuracy: %1.3f"%(mode,acc))
    else:
        acc1 = (total_correct_labels*1.0/total_labels)
        acc2 = (total_correct_cases*1.0/total_cases)
        print("%s accuracy: %1.3f\t%1.3f"%(mode,acc1, acc2))
    
    print("top-%d accuracy: %1.3f" %(args.k, total_correct_k*1.0/total_cases))
        
    return

def test(args):
    vocab = Vocab(args.dict_root, args.max_oovs)
    criterion = nn.NLLLoss()
    if args.load is None:
        print("Error: no model found")
        sys.exit()
    else:
        model = torch.load(args.load)
    if args.cuda:
        model.cuda()
    total_batches=0
    # args.val_root = args.test_root # to apply val function directly
    print_samples(model,vocab,args)
    val(model, vocab, args)
    return

def copy(args):
    import os
    import datetime
    from distutils.dir_util import copy_tree
    with open('performances.txt','a') as f:
        f.write(time_str)

    if not os.path.exsits("saves"):
        os.mkdir("saves")

    folder_dir = os.path.join('saves',args.time_str)
    os.mkdir(folder_dir)
    from_list = ['models/','packages/']
    for item in from_list:
        from_dir = item
        to_dir = os.path.join(folder_dir,item)
        copy_tree(from_dir, to_dir)
    print("Folders copied at %s" %folder_dir)
    if args.save_dir=='saves':
        args.save_dir = folder_dir
    with open(os.path.join(args.save_dir,"meta.txt"),'w') as f:
        f.write(str(args))
    return

def print_samples(model, vocab, args):
    data_loader = get_loader(args.val_root, args.dict_root, vocab, args.batch, 
                          args.single, shuffle=False)
    f = open(os.path.join(args.time_str,'samples.txt'),'a')
    for i, (inputs, lengths, labels, oovs) in enumerate(data_loader):
        model.eval()
        sources, queries, targets = inputs
        source_len, query_len, target_len, context_len= lengths
        if args.cuda:
            sources = sources.cuda()
            queries = queries.cuda()
            targets = targets.cuda()
        if args.single:
            outputs = model(sources,queries,lengths, targets) # [batch x seq x vocab]
        else:
            outputs, sim = model(sources,queries,lengths,targets)
        
        context = context_len[0]
        source = sources[:context]
        query = queries[0]
        target = targets[0][1:]
        output = outputs[0].max(1)[1]
        
        l1 = 'source: \n'+'\n'.join([vocab.tensor_to_string(src,oovs[0]) for src in source])
        l2 = 'query: '+ vocab.tensor_to_string(query,oovs[0])
        l3 = 'target: ' + vocab.tensor_to_string(target,oovs[0])
        l4 = 'output: ' + vocab.tensor_to_string(output.data,oovs[0])
        f.write(l1+'\n'+l2+'\n'+l3+'\n'+l4+'\n------------------------------------------\n\n')
    return

def main(args):
    if args.copy==True:
        copy(args)
    if args.mode=='train':
        print("Train mode")
        train(args)
    elif args.mode=='test':
        print("Test mode")
        test(args)
    else:
        print("Error: please specify --mode as 'train' or 'test'")
        
if __name__ == "__main__":
    main(args)
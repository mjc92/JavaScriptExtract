import numpy as np
import torch
from torch.autograd import Variable
import os
from collections import Counter
import torch
import glob

def write_log(string, args, filename):
    with open(os.path.join(args.save_dir,filename),'a') as f:
        f.write(string+'\n')

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def preprocess(line):
    line = line.replace("'"," ' ")
    line = line.replace('"',' " ')
    line = line.replace('.',' . ')
    line = line.replace(',',' , ')
    line = line.replace('+',' + ')
    line = line.replace('-',' - ')
    line = line.replace('=',' = ')
    line = line.replace('= =','==')
    line = line.replace('/',' / ')
    line = line.replace('*',' * ')
    line = line.replace('(',' ( ')
    line = line.replace(')',' ) ')
    line = line.replace('[',' [ ')
    line = line.replace(']',' ] ')
    line = line.replace('{',' { ')
    line = line.replace('}',' } ')
    line = line.replace(':',' : ')
    line = line.replace(';',' ; ')
    line = line.replace('  ',' ')
    line = line.strip()
    return line

def to_cuda(item, iscuda):
    if iscuda:
        return item.cuda()
    else:
        return item

def to_np(x):
    return x.data.cpu().numpy()

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def pack_padded(outputs, targets):
    """
    outputs: Variable, [b x seq x vocab]
    targets: Variable, [b x seq]
    """
    out_pack = []
    tar_pack = []
    target_lens = (targets>0).long().sum(1).data
    for i,length in enumerate(target_lens):
        out_pack.append(outputs[i,:length])
        tar_pack.append(targets[i,:length])
    return torch.cat(out_pack,0), torch.cat(tar_pack,0)

def smartprint(line):
    line = line.split('%NWL%')
    source = line[:-2]
    query = line[-2]
    idx = int(line[-1].strip())
    source[idx] = '* '+source[idx]
    print('\n'.join(source)+'\n'+query)
    return
from packages.vocab import Vocab
import torch
from packages.val import val
import sys

def test(args):
    vocab = Vocab(args.dict_root, args.max_oovs)
    if args.load is None:
        print("Error: no model found")
        sys.exit()
    else:
        model = torch.load(args.load)
    if args.cuda:
        model.cuda()
    total_batches = 0
    args.val_root = args.test_root
    val(model, vocab, args)
    return
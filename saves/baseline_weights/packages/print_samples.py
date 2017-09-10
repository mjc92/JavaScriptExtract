def print_samples(model, vocab, args):
    data_loader = get_loader(args.val_root, args.dict_root, vocab, args.batch,
                             args.single, shuffle=False)
    load_dir = '/'.join(args.load.split('/')[:2])
    f = open(os.path.join(load_dir, 'samples.txt'), 'a')
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
        if args.single:
            source = sources[0]
        else:
            context = context_len[0]
            source = sources[:context]
        query = queries[0]
        target = targets[0][1:]
        output = outputs[0].max(1)[1]
        if args.single:
            l1 = 'source: \n' + vocab.tensor_to_string(source, oovs[0])
        else:
            l1 = 'source: \n' + '\n'.join([vocab.tensor_to_string(src, oovs[0]) for src in source])
        l2 = 'query: ' + vocab.tensor_to_string(query, oovs[0])
        l3 = 'target: ' + vocab.tensor_to_string(target, oovs[0])
        l4 = 'output: ' + vocab.tensor_to_string(output.data, oovs[0])
        f.write(l1 + '\n' + l2 + '\n' + l3 + '\n' + l4 + '\n------------------------------------------\n\n')
    return
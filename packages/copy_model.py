def copy_model(args):
    import os
    import datetime
    from distutils.dir_util import copy_tree
    with open('performances.txt', 'a') as f:
        f.write(args.time_str)
    folder_dir = os.path.join('saves', args.time_str)
    os.mkdir(folder_dir)
    log_dir = os.path.join('logs', args.time_str)
    os.mkdir(log_dir)
    args.log_dir = log_dir
    # copy models and packages
    from_list = ['models/', 'packages/']
    for item in from_list:
        from_dir = item
        to_dir = os.path.join(folder_dir, item)
        copy_tree(from_dir, to_dir)
    print("Folders copied at %s" % folder_dir)
    if args.save_dir == 'saves':
        args.save_dir = folder_dir
    with open(os.path.join(args.save_dir, "meta.txt"), 'w') as f:
        f.write(str(args))
    return

def copy_model(args):
    import os
    import datetime
    import shutil
    from distutils.dir_util import copy_tree
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.time_str = time_str
    # with open('performances.txt', 'a') as f:
    #     f.write(args.time_str)
    if args.save_dir=='saves': # if default
        folder_dir = os.path.join('saves', args.time_str)
        log_dir = os.path.join('logs',args.time_str)
    else:
        folder_dir = os.path.join('saves',args.save_dir)
        log_dir = os.path.join('logs',args.save_dir)
    if not os.path.exists(folder_dir):
        os.mkdir(folder_dir)
    else:
        shutil.rmtree(folder_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        shutil.rmtree(log_dir)
    args.log_dir = log_dir
    # copy models and packages
    from_list = ['models/', 'packages/']
    for item in from_list:
        from_dir = item
        to_dir = os.path.join(folder_dir, item)
        copy_tree(from_dir, to_dir)
    print("Folders copied at %s" % folder_dir)
    args.save_dir = folder_dir
    with open(os.path.join(args.save_dir, "meta.txt"), 'w') as f:
        f.write(str(args))
    return

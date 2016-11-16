import sys, getopt, tarfile

msg_help = """python trainer.py <tar data>
-o <path to save ckpt>
-e <epoch to train, default 4>
-b <mini batch size, default 200>
-v (verbose output)
-p (show progress using carriage return)"""

def train(data_path, save_path, max_epoch=4, batchsize=100, lr_init=0.003, prog=False, stat=False):
    import data
    from chrecog.train import Trainer
    
    print('Ckpt save path : %s' % save_path)
    print('Reading tar file from %s...' % data_path)
    tar = tarfile.open(data_path, "r:*")
    label = data.get_label_from_tar(tar)
    
    trainer = Trainer(tar, label)
    trainer.init_session()
    trainer.train(max_epoch=max_epoch, batchsize=batchsize, lr_init=lr_init, prog=prog, stat=stat)
    trainer.save(save_path)

def main(argv):
    data_path = None
    save_path = None
    max_epoch = 4
    batchsize = 200
    lr_init = 0.003
    prog = False
    stat = False
    try:
        opts, args = getopt.gnu_getopt(argv,"hpvi:o:e:b:l:",["progress","verbose","input=","output=","epoch=","batch="])
    except getopt.GetoptError:
        print(msg_help)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(msg_help)
            sys.exit()
        elif opt in ("-p", "--progress"):
            prog = True
        elif opt in ("-v", "--verbose"):
            stat = True
        elif opt in ("-i", "--input"):
            data_path = arg
        elif opt in ("-o", "--output"):
            save_path = arg
        elif opt in ("-e", "--epoch"):
            max_epoch = int(arg)
        elif opt in ("-b", "--batch"):
            batchsize = int(arg)
        elif opt in ("-l"):
            lr_init = int(arg)
    
    if len(args) != 1:
        print(msg_help)
        sys.exit(2)
    
    data_path = args[0]
    
    if data_path is None or save_path is None:
        print(msg_help)
        sys.exit(2)
    
    train(data_path, save_path, max_epoch, batchsize, lr_init, prog, stat)

if __name__ == "__main__":
    main(sys.argv[1:])

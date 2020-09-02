import argparse
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')

parser = argparse.ArgumentParser(description='DanceNet AE_LSTM')
parser.add_argument('-t', '--train_dirs', default='../data/C_train_dirs.txt', type=str,
                        help='training data dirs file')
parser.add_argument('-m', '--model_save_dir', default='./model/C/motion_vae_model', type=str, help='model save dir')
parser.add_argument('-l', '--log_dir',default='./motion_log', type=str, help='log save dir')

parser.add_argument('--resume', default='false', type=str2bool,
                        help='whether continue training (must set model_load_dir)')



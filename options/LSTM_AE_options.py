import argparse
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')

parser = argparse.ArgumentParser(description='DanceNet AE_LSTM')
parser.add_argument('-t', '--train_dirs', default='../data/R_train_dirs.txt', type=str,
                        help='training data dirs file')
parser.add_argument('-m', '--model_save_dir', default='./model', type=str, help='model save dir')
parser.add_argument('-l', '--log_dir',default='./train_nn_log', type=str, help='log save dir')
parser.add_argument('--normalize_mode',default='minmax', type=str, help='normalize mode')


parser.add_argument('--model_load_dir',default='./model', type=str, help='model load dir')
parser.add_argument('--rnn_input_dim', default=32, type=int, help='rnn input dimension')
parser.add_argument('--rnn_unit_size', default=32, type=int, help='rnn unit size')
parser.add_argument('--acoustic_dim', default=16, type=int, help='acoustic feature dimension')
parser.add_argument('--temporal_dim', default=3, type=int, help='temporal feature dimension')
parser.add_argument('--motion_dim', default=63, type=int, help='motion feature dimension')
parser.add_argument('--time_step', default=120, type=int, help='time step length')
parser.add_argument('--batch_size', default=10, type=int, help='minibatch size')
parser.add_argument('--learning_rate', default=1e-3, type=float, help='learning rate')
parser.add_argument('--extr_loss_threshold', default=0.045, type=float, help='extr loss threshold')
parser.add_argument('--epoch_size', default=1000, type=int, help='epoch num')
parser.add_argument('--dense_dim', default=24, type=int, help='dense layer output dimension')
parser.add_argument('--lstm_output_dim', default=32, type=int, help='lstm output dimension')
parser.add_argument('--reduced_size', default=10, type=int, help='latent size')



parser.add_argument('--overlap', default='true', type=str2bool,
                        help='whether use overlap data')
parser.add_argument('--use_mask', default='true', type=str2bool,
                        help='whether use masking layer')
parser.add_argument('--resume', default='false', type=str2bool,
                        help='whether continue training (must set model_load_dir)')



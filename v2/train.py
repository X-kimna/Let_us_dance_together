from VAE_LSTM_FIX_model import  VAE_LSTM_FIX_model
from options.LSTM_FixedVAE_options import parser
if __name__=='__main__':
    args = parser.parse_args()
    train_dirs = []
    with open(args.train_dirs,'r')as f:
        for line in f.readlines():
            train_dirs.append(line[:-1])

    Model=VAE_LSTM_FIX_model(
                 train_file_list=train_dirs,
                 model_save_dir=args.model_save_dir,
                 model_load_dir=args.model_load_dir,
                 log_dir=args.log_dir,
                 motion_vae_ckpt_dir=args.motion_vae_ckpt_dir,
                 music_vae_ckpt_dir=args.music_vae_ckpt_dir,
                 rnn_unit_size=args.rnn_unit_size,
                 acoustic_dim=args.acoustic_dim,
                 temporal_dim=args.temporal_dim,
                 motion_dim=args.motion_dim,
                 time_step=args.time_step,
                 batch_size=args.batch_size,
                 learning_rate=args.learning_rate,
                 extr_loss_threshold=args.extr_loss_threshold,
                 overlap=args.overlap,
                 epoch_size=args.epoch_size,
                 use_mask=args.use_mask,
                 normalize_mode=args.normalize_mode)

    Model.train(resume=args.resume)
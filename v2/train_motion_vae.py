from MotionVae import  MotionVae
from options.motion_vae_options import parser
if __name__=="__main__":
    args = parser.parse_args()
    train_dirs = []
    with open(args.train_dirs, 'r')as f:
        for line in f.readlines():
            train_dirs.append(line[:-1])

    Model = MotionVae(model_save_dir=args.model_save_dir,
                 log_dir=args.log_dir,
                 train_file_list=train_dirs)
    Model.init_dataset(normalize_mode=args.normalize_mode)
    Model.train(resume=args.resume)
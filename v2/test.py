from VAE_LSTM_FIX_model import VAE_LSTM_FIX_model

if __name__ == '__main__':
    train_dirs = []
    with open('../data/R_train_dirs.txt', 'r')as f:
        for line in f.readlines():
            train_dirs.append(line[:-1])
    test_dirs=[
        "../data/DANCE_R_10"
    ]
    Model = VAE_LSTM_FIX_model(
        train_file_list=train_dirs,
        model_save_dir='./good_result/W/model',
        model_load_dir='./good_result/W/model',
        log_dir='./train_nn_log',
        motion_vae_ckpt_dir='./good_result/W/motion_vae_model/stock2.model-999',
        music_vae_ckpt_dir='./good_result/W/music_vae_model/stock2.model-769',
        rnn_unit_size=64,
        acoustic_dim=16,
        temporal_dim=3,
        motion_dim=63,
        time_step=120,
        batch_size=10,
        learning_rate=1e-4,
        extr_loss_threshold=6e-4,
        overlap=True,
        epoch_size=1000,
        use_mask=True)

    for test_file in test_dirs:
        Model.predict(test_file,'./result')

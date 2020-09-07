from data_prepare.feature_extract import  Music
from VAE_LSTM_FIX_TCV_model import VAE_LSTM_FIX_TCV_model
import librosa
import json
import numpy as np
import os
from data_prepare.visualize import  draw_predict
from visualization.threeDPoints2Bvh import smartbody_skeleton
def getStandardFrames(frames):
    new_frames = np.zeros([len(frames), 21, 3])
    for i in range(len(frames)):
        # Hips
        new_frames[i][0][0] = frames[i][2][0] * -1
        new_frames[i][0][1] = frames[i][2][1]
        new_frames[i][0][2] = frames[i][2][2]
        # RightHip
        new_frames[i][1][0] = frames[i][16][0] * -1
        new_frames[i][1][1] = frames[i][16][1]
        new_frames[i][1][2] = frames[i][16][2]

        # RightKnee
        new_frames[i][2][0] = frames[i][17][0] * -1
        new_frames[i][2][1] = frames[i][17][1]
        new_frames[i][2][2] = frames[i][17][2]
        # RightAnkle
        new_frames[i][3][0] = frames[i][18][0] * -1
        new_frames[i][3][1] = frames[i][18][1]
        new_frames[i][3][2] = frames[i][18][2]

        # LeftHip
        new_frames[i][4][0] = frames[i][7][0] * -1
        new_frames[i][4][1] = frames[i][7][1]
        new_frames[i][4][2] = frames[i][7][2]
        # LeftKnee
        new_frames[i][5][0] = frames[i][8][0] * -1
        new_frames[i][5][1] = frames[i][8][1]
        new_frames[i][5][2] = frames[i][8][2]
        # LeftAnkle
        new_frames[i][6][0] = frames[i][9][0] * -1
        new_frames[i][6][1] = frames[i][9][1]
        new_frames[i][6][2] = frames[i][9][2]

        temp1 = [(frames[i][12][0] + frames[i][3][0]) / 2, (frames[i][12][1] + frames[i][3][1]) / 2,
                 (frames[i][12][2] + frames[i][3][2]) / 2]
        temp2 = [(frames[i][1][0] + frames[i][0][0]) / 2, (frames[i][1][1] + frames[i][0][1]) / 2,
                 (frames[i][1][2] + frames[i][0][2]) / 2]

        # Spine
        new_frames[i][7][0] = (temp1[0] + frames[i][2][0]) / 2 * -1
        new_frames[i][7][1] = (temp1[1] + frames[i][2][1]) / 2
        new_frames[i][7][2] = (temp1[2] + frames[i][2][2]) / 2
        # Thorax
        new_frames[i][8][0] = temp1[0] * -1
        new_frames[i][8][1] = temp1[1]
        new_frames[i][8][2] = temp1[2]

        # Neck
        new_frames[i][9][0] = (temp1[0] + (temp2[0] - temp1[0]) * 0.5) * -1
        new_frames[i][9][1] = (temp1[1] + (temp2[1] - temp1[1]) * 0.5)
        new_frames[i][9][2] = (temp1[2] + (temp2[2] - temp1[2]) * 0.5)
        # Head
        new_frames[i][10][0] = (temp1[0] + (temp2[0] - temp1[0]) * 1.3) * -1
        new_frames[i][10][1] = (temp1[1] + (temp2[1] - temp1[1]) * 1.3)
        new_frames[i][10][2] = (temp1[2] - (temp2[2] - temp1[2]) * 0.5)

        # LeftShoulder
        new_frames[i][11][0] = frames[i][3][0] * -1
        new_frames[i][11][1] = frames[i][3][1]
        new_frames[i][11][2] = frames[i][3][2]

        # LeftElbow
        new_frames[i][12][0] = frames[i][4][0] * -1
        new_frames[i][12][1] = frames[i][4][1]
        new_frames[i][12][2] = frames[i][4][2]
        # LeftWrist
        new_frames[i][13][0] = frames[i][5][0] * -1
        new_frames[i][13][1] = frames[i][5][1]
        new_frames[i][13][2] = frames[i][5][2]

        # RightShoulder
        new_frames[i][14][0] = frames[i][12][0] * -1
        new_frames[i][14][1] = frames[i][12][1]
        new_frames[i][14][2] = frames[i][12][2]
        # RightElbow
        new_frames[i][15][0] = frames[i][13][0] * -1
        new_frames[i][15][1] = frames[i][13][1]
        new_frames[i][15][2] = frames[i][13][2]

        # RightWrist
        new_frames[i][16][0] = frames[i][14][0] * -1
        new_frames[i][16][1] = frames[i][14][1]
        new_frames[i][16][2] = frames[i][14][2]

        # LeftWristEndSite
        new_frames[i][17][0] = frames[i][6][0] * -1
        new_frames[i][17][1] = frames[i][6][1]
        new_frames[i][17][2] = frames[i][6][2]

        # RightWristEndSite
        new_frames[i][18][0] = frames[i][15][0] * -1
        new_frames[i][18][1] = frames[i][15][1]
        new_frames[i][18][2] = frames[i][15][2]

        # LeftToe
        new_frames[i][19][0] = (frames[i][11][0] + frames[i][10][0]) / 2 * -1
        new_frames[i][19][1] = (frames[i][11][1] + frames[i][10][1]) / 2
        new_frames[i][19][2] = (frames[i][11][2] + frames[i][10][2]) / 2

        # RightToe
        new_frames[i][20][0] = (frames[i][20][0] + frames[i][19][0]) / 2 * -1
        new_frames[i][20][1] = (frames[i][20][1] + frames[i][19][1]) / 2
        new_frames[i][20][2] = (frames[i][20][2] + frames[i][19][2]) / 2
    return new_frames

def smooth(a, WSZ):
    out0 = np.convolve(a, np.ones(WSZ, dtype=int), 'valid') / WSZ
    r = np.arange(1, WSZ - 1, 2)
    start = np.cumsum(a[:WSZ - 1])[::2] / r
    stop = (np.cumsum(a[:-WSZ:-1])[::2] / r)[::-1]
    return np.concatenate((start, out0, stop))


def smooth_skeleton(motion):
    WSZ = 3
    skeletons_num = motion.shape[1]
    skeletons = np.hsplit(motion, skeletons_num)
    cur_skeleton = np.reshape(skeletons[0], (-1, 3))
    # print(cur_skeleton.shape)
    x_seq = np.split(cur_skeleton, 3, axis=1)[0]
    x_seq = np.reshape(x_seq, -1)

    y_seq = np.split(cur_skeleton, 3, axis=1)[1]
    y_seq = np.reshape(y_seq, -1)

    z_seq = np.split(cur_skeleton, 3, axis=1)[2]
    z_seq = np.reshape(z_seq, -1)

    x_smooth = smooth(x_seq, WSZ)
    y_smooth = smooth(y_seq, WSZ)
    z_smooth = smooth(z_seq, WSZ)
    x_smooth = np.array(x_smooth)
    smooth_result = np.column_stack((x_smooth, y_smooth, z_smooth))

    for i in range(1, motion.shape[1]):
        cur_skeleton = np.reshape(skeletons[i], (-1, 3))
        # print(cur_skeleton.shape)
        x_seq = np.split(cur_skeleton, 3, axis=1)[0]
        x_seq = np.reshape(x_seq, -1)

        y_seq = np.split(cur_skeleton, 3, axis=1)[1]
        y_seq = np.reshape(y_seq, -1)

        z_seq = np.split(cur_skeleton, 3, axis=1)[2]
        z_seq = np.reshape(z_seq, -1)

        x_smooth = smooth(x_seq, WSZ)
        y_smooth = smooth(y_seq, WSZ)
        z_smooth = smooth(z_seq, WSZ)
        x_smooth = np.array(x_smooth)
        # print(x_smooth.shape)
        x = np.linspace(1, 5050, 5050)  # X轴数据
        tmp = np.column_stack((x_smooth, y_smooth, z_smooth))
        if i == 1:
            smooth_result = np.stack((smooth_result, tmp), axis=1)
        else:
            tmp_ = tmp[:, np.newaxis, :]
            smooth_result = np.concatenate((smooth_result, tmp_), axis=1)

    return smooth_result
hop_length = 512
window_length = hop_length * 2
fps = 25
spf = 0.04  # 40 ms
sample_rate = 44100  #
resample_rate = hop_length * fps



music_type='C'
music_dir= '../music/%s'%music_type
music_name='Havana'
music_path=os.path.join(music_dir,music_name+'.mp3')
duration =librosa.get_duration(filename=music_path)


music = Music(music_path, sr=resample_rate, start=0, duration=duration) # 25fps
acoustic_features, temporal_indexes = music.extract_features()  # 16 dim
acoustic_features_path = os.path.join(music_dir, music_name+"_acoustic_features.npy")
temporal_indexes_path = os.path.join(music_dir, music_name+"_temporal_features.npy")
np.save(acoustic_features_path, acoustic_features)
np.save(temporal_indexes_path, temporal_indexes)



train_dirs = []
with open('../data/%s_train_dirs.txt'%music_type, 'r')as f:
    for line in f.readlines():
        train_dirs.append(line[:-1])

Model = VAE_LSTM_FIX_TCV_model(
    train_file_list=train_dirs,
    model_save_dir='./model/%s/model'%music_type,
    model_load_dir='./model/%s/model'%music_type,
    log_dir='./train_nn_log',
    motion_vae_ckpt_dir='./model/%s/motion_vae_model/stock2.model-999'%music_type,
    music_vae_ckpt_dir='./model/%s/music_vae_model/stock2.model-999'%music_type,
    rnn_unit_size=32,
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
result_save_dir= '../result/%s'%music_type
Model.predict_from_music(acoustic_features, temporal_indexes,music_name,result_save_dir=result_save_dir)
motion_path=os.path.join(result_save_dir,music_name+'.json')

draw_predict(motion_path, result_save_dir,music_name,temporal_indexes_path,music_path)

bvh_path=music_path=os.path.join(result_save_dir,music_name+'.bvh')

with open(motion_path, 'r') as fin:
        data = json.load(fin)

frames = np.array(data['skeletons'])
frames = getStandardFrames(frames)
smartbody_skeleton = smartbody_skeleton.SmartBodySkeleton()
smartbody_skeleton.poses2bvh(frames, output_file=bvh_path)




import numpy as np
import cv2

import json
import subprocess
import math
from moviepy.editor import *
from moviepy.video import VideoClip
from data_prepare.feature_extract import rotate_skeleton
from data_prepare.feature_extract import load_start_end_frame_num, load_skeleton

music_path='../music/R/DANCE_R_10.mp3'

v1_path='./comparision/DANCE_R_10_1.json'  # AE
v2_path='./comparision/DANCE_R_10_2.json'  # vae fixed
v3_path='./comparision/DANCE_R_10_3.json'  # vae
v4_path='./comparision/DANCE_R_10_4.json'  # tcv



fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

with_rotate = True
to_video = True
show_ground_truth = True
show_pred = True
show_bones = 21
skeleton_target_color = (0, 0, 0)  # Line color
skeleton_pred_color = (0, 0, 0)  # Line color
bone_pred_color = (255, 0, 0)
bone_target_color = (0, 0, 255)
fps = 25
scale = 3
CANVAS_SIZE = (2000, 3 * 200, 3)
root_dir = 'F:/srtp/$RTP/Music-to-Dance-Motion-Synthesis//DANCE_R_10//'

audio_path = root_dir + 'audio.mp3'
target_path = root_dir + 'skeletons.json'
pred_video = root_dir + 'output.mp4'
pred_video2 = root_dir + 'ouput_music.mp4'
config_path = root_dir + 'config.json'
tempo_path = root_dir + 'temporal_features.npy'



def draw_hints(cvs):

    cv2.putText(cvs, 'LSTM-AE', (900 // 4 - scale * 25, scale * 15), cv2.FONT_ITALIC, scale * 0.3,
                bone_pred_color, 1)
    cv2.putText(cvs, 'LSTM-Fixed-VAE', (900 * 3 // 4 - scale * 25, scale * 15), cv2.FONT_ITALIC, scale * 0.3,
                bone_target_color, 1)

    cv2.putText(cvs, 'LSTM-VAE', (900 * 5 // 4 - scale * 25, scale * 15), cv2.FONT_ITALIC, scale * 0.3,
                bone_target_color, 1)

    cv2.putText(cvs, 'LSTM-Fixed-VAE-TC', (900 * 7 // 4 - scale * 25, scale * 15), cv2.FONT_ITALIC, scale * 0.3,
                bone_target_color, 1)

def draw_beat(cvs, this_beat):
    cv2.putText(cvs, 'frame:' + str(int(this_beat[0])), (scale * 3, CANVAS_SIZE[1] - scale * 3), cv2.FONT_ITALIC,
                scale * 0.3, (0, 0, 255), 1)
    cv2.putText(cvs, 'beat:' + str(int(this_beat[1])), (scale * 3, CANVAS_SIZE[1] - scale * 9), cv2.FONT_ITALIC,
                scale * 0.3, (0, 0, 255), 1)
    cv2.putText(cvs, 'in beat frame:' + str(int(this_beat[2])), (scale * 3, CANVAS_SIZE[1] - scale * 15),
                cv2.FONT_ITALIC, scale * 0.3,
                (0, 0, 255), )


def draw_skeleton_number(cvs, frame):
    for j in range(show_bones):
        cv2.putText(cvs, str(j), (int(frame[j][0]), (CANVAS_SIZE[1] - int(frame[j][1]))), cv2.FONT_ITALIC, scale * 0.3,
                    (0, 0, 255), 1)


def draw_skeleton(cvs, frame, bone_color, skeleton_color, ):
    for j in range(show_bones):
        cv2.circle(cvs, (int(frame[j][0]), int(frame[j][1])), radius=scale * 3, thickness=-1, color=bone_color)
    cv2.line(cvs, (int(frame[0][0]), int(frame[0][1])), (int(frame[1][0]), int(frame[1][1])), skeleton_color, 2)
    cv2.line(cvs, (int((frame[0][0] + frame[1][0]) / 2), int((frame[0][1] + frame[1][1]) / 2)),
             (int((frame[3][0] + frame[12][0]) / 2), int((frame[3][1] + frame[12][1]) / 2)), skeleton_color, 2)
    cv2.line(cvs, (int(frame[3][0]), int(frame[3][1])),
             (int((frame[3][0] + frame[12][0]) / 2), int((frame[3][1] + frame[12][1]) / 2)), skeleton_color, 2)
    cv2.line(cvs, (int(frame[3][0]), int(frame[3][1])), (int(frame[4][0]), int(frame[4][1])), skeleton_color, 2)
    cv2.line(cvs, (int(frame[4][0]), int(frame[4][1])), (int(frame[5][0]), int(frame[5][1])), skeleton_color, 2)
    cv2.line(cvs, (int(frame[5][0]), int(frame[5][1])), (int(frame[6][0]), int(frame[6][1])), skeleton_color, 2)
    cv2.line(cvs, (int(frame[12][0]), int(frame[12][1])),
             (int((frame[3][0] + frame[12][0]) / 2), int((frame[3][1] + frame[12][1]) / 2)), skeleton_color, 2)
    cv2.line(cvs, (int(frame[12][0]), int(frame[12][1])), (int(frame[13][0]), int(frame[13][1])), skeleton_color, 2)
    cv2.line(cvs, (int(frame[13][0]), int(frame[13][1])), (int(frame[14][0]), int(frame[14][1])), skeleton_color, 2)
    cv2.line(cvs, (int(frame[14][0]), int(frame[14][1])), (int(frame[15][0]), int(frame[15][1])), skeleton_color, 2)
    cv2.line(cvs, (int(frame[2][0]), int(frame[2][1])),
             (int((frame[3][0] + frame[12][0]) / 2), int((frame[3][1] + frame[12][1]) / 2)), skeleton_color, 2)
    cv2.line(cvs, (int(frame[2][0]), int(frame[2][1])), (int(frame[7][0]), int(frame[7][1])), skeleton_color, 2)
    cv2.line(cvs, (int(frame[7][0]), int(frame[7][1])), (int(frame[8][0]), int(frame[8][1])), skeleton_color, 2)
    cv2.line(cvs, (int(frame[8][0]), int(frame[8][1])), (int(frame[9][0]), int(frame[9][1])), skeleton_color, 2)
    cv2.line(cvs, (int(frame[9][0]), int(frame[9][1])),
             (int((frame[10][0] + frame[11][0]) / 2), int((frame[10][1] + frame[11][1]) / 2)), skeleton_color, 2)
    cv2.line(cvs, (int(frame[10][0]), int(frame[10][1])), (int(frame[11][0]), int(frame[11][1])), skeleton_color, 2)
    cv2.line(cvs, (int(frame[2][0]), int(frame[2][1])), (int(frame[16][0]), int(frame[16][1])), skeleton_color, 2)
    cv2.line(cvs, (int(frame[16][0]), int(frame[16][1])), (int(frame[17][0]), int(frame[17][1])), skeleton_color, 2)
    cv2.line(cvs, (int(frame[17][0]), int(frame[17][1])), (int(frame[18][0]), int(frame[18][1])), skeleton_color, 2)
    cv2.line(cvs, (int(frame[18][0]), int(frame[18][1])),
             (int((frame[19][0] + frame[20][0]) / 2), int((frame[19][1] + frame[20][1]) / 2)), skeleton_color, 2)
    cv2.line(cvs, (int(frame[19][0]), int(frame[19][1])), (int(frame[20][0]), int(frame[20][1])), skeleton_color, 2)



if __name__ == '__main__':
    with open(v1_path, 'r') as v1, open(v2_path, 'r') as v2, open(v3_path, 'r') as v3,open(v4_path, 'r') as v4:

        v1_data = json.load(v1)
        v2_data = json.load(v2)
        v3_data = json.load(v3)
        v4_data = json.load(v4)

        frames1 = np.array(v1_data['skeletons'])
        frames2 = np.array(v2_data['skeletons'])
        frames3 = np.array(v3_data['skeletons'])
        frames4 = np.array(v4_data['skeletons'])

        min_len =  min(min(len(frames1), min(len(frames2), len(frames3))),len(frames4))
        frames1 = frames1[:min_len]
        frames2 = frames2[:min_len]
        frames3 = frames3[:min_len]
        frames4 = frames4[:min_len]


        frames1[:, :, 0] *= scale
        frames1[:, :, 1] *= scale
        frames1[:, :, 0] += 900 // 4
        frames1[:, :, 1] += CANVAS_SIZE[1] // 2

        frames2[:, :, 0] *= scale
        frames2[:, :, 1] *= scale
        frames2[:, :, 0] += 900 * 3 // 4
        frames2[:, :, 1] += CANVAS_SIZE[1] // 2

        frames3[:, :, 0] *= scale
        frames3[:, :, 1] *= scale
        frames3[:, :, 0] += 900 * 5 // 4
        frames3[:, :, 1] += CANVAS_SIZE[1] // 2

        frames4[:, :, 0] *= scale
        frames4[:, :, 1] *= scale
        frames4[:, :, 0] += 900 * 7 // 4
        frames4[:, :, 1] += CANVAS_SIZE[1] // 2

        tempo = np.load(tempo_path)
        video = cv2.VideoWriter(pred_video, fourcc, fps, (CANVAS_SIZE[0], CANVAS_SIZE[1]), 1)
        image_seq = []
        for i in range(len(frames1)):
            cvs = np.ones((CANVAS_SIZE[1], CANVAS_SIZE[0], CANVAS_SIZE[2]))
            cvs[:, :, :] = 255

            this_beat = tempo[i]
            frame1 = frames1[i]
            draw_skeleton(cvs, frame1, bone_pred_color, skeleton_pred_color)
            frame2 = frames2[i]
            draw_skeleton(cvs, frame2, bone_target_color, skeleton_target_color)

            frame3 = frames3[i]
            draw_skeleton(cvs, frame3, bone_pred_color, skeleton_pred_color)
            frame4 = frames4[i]
            draw_skeleton(cvs, frame4, bone_target_color, skeleton_target_color)



            ncvs = np.flip(cvs, 0).copy()

            draw_skeleton_number(ncvs, frame1)
            draw_skeleton_number(ncvs, frame2)
            draw_skeleton_number(ncvs, frame3)
            draw_skeleton_number(ncvs, frame4)
            draw_beat(ncvs, this_beat)
            draw_hints(ncvs)
            if not to_video:
                cv2.imshow('canvas', ncvs)
                cv2.waitKey(0)

            # image_seq = ncvs if image_seq is None else np.append(image_seq,ncvs)
            video.write(np.uint8(ncvs))
        # clip = ImageSequenceClip(image_seq,fps=fps)
        # clip.write_videofile(filename=video_path)
        video.release()
        start, end = load_start_end_frame_num(config_path)
        audio = AudioFileClip(audio_path)

        sub = audio.subclip(start / fps, end / fps)
        print('Analyzed the audio, found a period of %.02f seconds' % sub.duration)
        video = VideoFileClip(pred_video, audio=False)
        video = video.set_audio(sub)
        video.write_videofile(pred_video2)
        pass
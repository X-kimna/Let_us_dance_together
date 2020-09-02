import glob
import shutil
import os
count=0
dir_list=None
for root, dirs, files in os.walk('F:/srtp/$RTP/Music-to-Dance-Motion-Synthesis/'):
    if(count==0):
        dir_list=dirs
        break

for dir in dir_list:
    #os.makedirs('./'+dir)
    path=os.path.join('F:/srtp/$RTP/Music-to-Dance-Motion-Synthesis/',dir)
    shutil.copy(os.path.join(path,'audio.mp3'),os.path.join('./'+dir,'audio.mp3'),)
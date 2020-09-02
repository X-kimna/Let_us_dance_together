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
    shutil.copy(os.path.join(path,'config.json'),os.path.join('./'+dir,'config.json'),)
    shutil.copy(os.path.join(path, 'skeletons.json'), os.path.join('./' + dir, 'skeletons.json'), )
    shutil.copy(os.path.join(path, 'skeletons.json'), os.path.join('./' + dir, 'skeletons.json'), )
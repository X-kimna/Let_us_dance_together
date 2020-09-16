# readme

## Dependencies

tensorflow-gpu           1.6.0

librosa                  0.6.0

Keras-Preprocessing      1.1.2

numpy                    1.19.1

opencv-contrib-python    4.3.0.36

opencv-python            4.3.0.36

sklearn                  0.0

scikit-video             1.1.11

scikit-image             0.17.2

## dataset

You can download the origin dataset from, we also provide preprocessed data in `./data`: 

https://github.com/mhy12345/Music-to-Dance-Motion-Synthesis

## Details

We provide three version training mode:

1. v1: LSTM & autoencoder
2. v2:  LSTM & fixed VAE model  (only training lstm, the vae is pretrained)
3. v3:  LSTM & VAE model (training together)
4. v4: LSTM & fixed VAE model  & Temporal Convolution

## Train

#### V1

```
cd v1
python train.py
```

You can change the training details by your self,  (see the `./options/AE_options.py`) , for example:

```
cd v1
python train.py --train_dirs=../data/T_train_dirs.txt --learning_rate=1e-4 --epoch_size=500 --normalize_mode=standard
```

#### V2

You can train motionVAE and musicVAE by yourself:

```
cd v2
python train_motion_vae.py
python train_music_vae.py
```

We also provide pretrained VAE models :

**baiduYun**

链接：https://pan.baidu.com/s/1VxklDyWodBikT-DM9W-pFQ 
提取码：pbso

**Google Drive**

https://drive.google.com/drive/folders/1wC8vxr8t7rCedVA1g2g4YZ1E5xhQobZr?usp=sharing

Download the files in `VAE` and put them at `v2/model`

then :

```
cd v2
python train.py
```

You can also change the training details by your self,  (see the `./options/AE_options.py`) , for example:

```
cd v2
python train.py --train_dirs=../data/T_train_dirs.txt --learning_rate=1e-4 --motion_vae_ckpt_dir='./model/W/motion_vae_model/stock2.model-999'
```

please carefully set  --motion_vae_ckpt_dir='./motion_vae_model/stock2.model-**xxx**'  and  --motion_vae_ckpt_dir='./music_vae_model/stock2.model-**xxx**'

#### V3

```
cd v3
python train.py
```

You can also change the training details by yourself,  (see the `./options/AE_options.py`) , for example:

```
cd v3
python train.py --train_dirs=../data/T_train_dirs.txt --learning_rate=1e-4 --epoch_size=500 --normalize_mode=standard
```

#### V4

 the same as v2

## Test

for **v1,v2,v3,v4**, you can run the following `.py` files for testing

`./v1/test.py`

`./v2/test.py`

`./v3/test.py`

`./v4/test.py`

## Visualization

we provide `generate_dance_from_music.py` in `./v2` for you to generate a video that contains music and dance, for more vivid results, you can also get the  `.bvh` file. 

The `.bvh` file is used to drive the rigged 3D model to move, we use a simple application **liveAnimation** to do this work, more details can be find :



**You can use the applications in https://github.com/oneThousand1000/3DPointsMotionVisualization to visualize `.bvh` files:**

Use the application in `./liveAnimation/LiveAnimation_110/LiveAnimation` (**Notice: you need to install `./liveAnimation/xnafx40_redist.msi` firstly**), load the `.pmx`  model in `./model`, then import the `.bvh` file mentioned aboved, finally, enjoy the dance~ :)

![img](/images/1.png)



## Pretrained-model

You can get the pretrained model of **v3** from: 

**baiduYun**

链接：https://pan.baidu.com/s/1VxklDyWodBikT-DM9W-pFQ 
提取码：pbso

**Google Drive**

https://drive.google.com/drive/folders/1wC8vxr8t7rCedVA1g2g4YZ1E5xhQobZr?usp=sharing

## Result

You can get the final results of **v3** from: 

**baiduYun**

链接：https://pan.baidu.com/s/1lAHzNf4dJj6PNh-ZhSG5XA 
提取码：sk7n

**Google Drive**

https://drive.google.com/drive/folders/1gU2DN0RYrphlHBmLh-IXjw9wlO-qcRfi?usp=sharing

We will upload more videos in the future.

## contact

e-mail: onethousand@zju.edu.cn

## reference

TCN:

https://github.com/YuanTingHsieh/TF_TCN

## Paper

```
@inproceedings{tang2018dance,
	title={Dance with Melody: An LSTM-autoencoder Approach to Music-oriented Dance Synthesis},
	author={Tang, Taoran and Jia, Jia and Mao, Hanyang},
	booktitle={2018 ACM Multimedia Conference on Multimedia Conference},
	pages={1598--1606},
	year={2018},
	organization={ACM}
}
@inproceedings{tang2018anidance,
	title={AniDance: Real-Time Dance Motion Synthesize to the Song},
	author={Tang, Taoran and Mao, Hanyang and Jia, Jia},
	booktitle={2018 ACM Multimedia Conference on Multimedia Conference},
	pages={1237--1239},
	year={2018},
	organization={ACM}
}
@misc{bai2018trellis,
    title={Trellis Networks for Sequence Modeling},
    author={Shaojie Bai and J. Zico Kolter and Vladlen Koltun},
    year={2018},
    eprint={1810.06682},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```


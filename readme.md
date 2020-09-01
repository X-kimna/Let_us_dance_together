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

You can download the dataset from: 

https://github.com/mhy12345/Music-to-Dance-Motion-Synthesis

## Train

First of all, please modify the `train_dirs.txt` to prepare the train dataset.

for **v1,v2,v4**, you can run the following `.py` files for training 

`./v1/train.py`

`./v2/train.py`

`./v4/train.py`

for **v3**,  you must run the `./v3/train_motion_vae.py` and `./v3/train_music_vae.py` to get the pretrained VAE, then  run the `./v3/train.py` .

For the details of the network, please modify the model by yourself (e.g., the number of epoches, learning_rate, the model save dir)

## Test

for **v1,v2,v3,v4**, you can run the following `.py` files for testing

`./v1/test.py`

`./v2/test.py`

`./v3/test.py`

`./v4/test.py`

## Visualization

we provide `generate_dance_from_music.py` in `./v3` for you to generate a video that contains music and dance, for more vivid results, you can also get the  `.bvh` file.



**You can use the applications in https://github.com/oneThousand1000/3DPointsMotionVisualization to visualize `.bvh` files:**

Use the application in `./liveAnimation/LiveAnimation_110/LiveAnimation` (**Notice: you need to install `./liveAnimation/xnafx40_redist.msi` firstly**), load the `.pmx`  model in `./model`, then import the `.bvh` file mentioned aboved, finally, enjoy the dance~ :)

![img](/images/1.png)



## Pretrained-model

You can get the pretrained model of **v3** from: 

**baiduYun**

链接：https://pan.baidu.com/s/1VxklDyWodBikT-DM9W-pFQ 
提取码：pbso

## Result

You can get the final results of **v3** from: 

**baiduYun**

链接：https://pan.baidu.com/s/1lAHzNf4dJj6PNh-ZhSG5XA 
提取码：sk7n

## contact

e-mail: onethousand@zju.edu.cn

## Paper

我们的工作（v1，v2部分）基于以下论文

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
```


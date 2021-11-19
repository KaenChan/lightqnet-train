# LightQNet


This is a demo code of training and testing [LightQNet] using Tensorflow.

Uncertainty Losses:
+ IDQ loss
+ PCNet loss

Uncertainty Networks:
+ MobileNetv3-Small
+ Resnet18

##  Usage

### Preprocessing

Download the MS-Celeb-1M dataset from 1 or 2:
1. insightface, https://github.com/deepinsight/insightface/wiki/Dataset-Zoo
2. face.evoLVe.PyTorch, https://github.com/ZhaoJ9014/face.evoLVe.PyTorch#Data-Zoo) 

Decode it using the code:
https://github.com/deepinsight/insightface/blob/master/recognition/common/rec2image.py

### Training
1. Download the base model [ResFace64](https://drive.baidu.com/open?id=1MiC_qCj5GFidWLtON9ekClOCJu6dPHT4) and unzip the files under ```log/resface64```.

2. Modify the configuration files under ```configfig/``` folder.

4. Start the training:

    ``` Shell
    python train_idq.py configfig/resface64_msarcface_with_mbv3_small_idq.py
    ```
   
### Testing

We use lfw.bin, cfp_fp.bin, etc. from ms1m-retinaface-t1 as the test dataset.

``` Shell
python evaluation/verification_risk_fnmr.py
```
  
### Freeze and Deploy

Freeze

``` Shell
python freeze_idq.py --model_dir log/resface64_mbv3/20210128-150935
```

Deployment code

https://github.com/KaenChan/lightqnet

## Pre-trained Model

#### ResFace64
| Method | Download |
| ------ |--- |
|Base Mode| [Baidu Drive](https://pan.baidu.com/s/1ACjDBxA0tWFXs70J4dDv2A) PW:v800|
|Mobilenetv3-small + IDQ loss + Distillation | [Baidu Drive](https://pan.baidu.com/s/1li3q2XEFg_Axv-asYBiSYw) PW:3zgi|

#### Reference
If you find this repo useful, please consider citing:
```
@article{chen2021lightqnet,
  title={LightQNet: Lightweight Deep Face Quality Assessment for Risk-Controlled Face Recognition},
  author={Chen, Kai and Yi, Taihe and Lv, Qi},
  journal={IEEE Signal Processing Letters},
  volume={28},
  pages={1878--1882},
  year={2021},
  publisher={IEEE}
}
```

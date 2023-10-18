# MccSTN
## MccSTN: A Multi-scale Contrast and Fine-grained Feature Fusion Networks for Subject-driven Style Transfer
![Image_start](https://github.com/haizhu12/MccSTN/assets/93024130/920b7f49-3b25-40fd-975e-d4df7311a471)

### Abstract

Stylistic transformation of artistic images is an important part of the current image processing field. In order to access the aesthetic-artistic expression of style images, recent research has applied attention mechanisms to the field of style transfer. This approach transforms stylised images into tokens by calculating attention and then migrates the artistic style of the image through a decoder. However, due to the very low semantic similarity between the original image and the style image, this results in many fine-grained style features being discarded. This can lead to discordant artefacts or obvious artefacts. To address this problem, we propose MccSTN, a novel style representation and transfer framework that can be adapted to existing arbitrary image style transfers. Specifically, we first introduce a feature fusion module (Mccformer) to fuse aesthetic features in style images with fine-grained features in content images. Feature maps are obtained through Mccformer. The feature map is then fed into the decoder to get the image we want. In order to lighten the model and train it quickly, we consider the relationship between specific styles and the overall style distribution. We introduce a multi-scale augmented contrast module that learns style representations from a large number of image pairs. Comparative experiments show that our approach outperforms the state-of-the-art existing work.

### An overview of our Saiency-aware Noise Blending.

![main](https://github.com/haizhu12/MccSTN/assets/93024130/8b1b1dcd-b83f-4439-9441-a6818509235d)


### Preparation

#### Environment


from [pytorch](https://pytorch.org/)  


To set up their environment, please run:
```
python=3.8
pytorch=1.9.1
```

Clone this repo:
```
git clone https://github.com/haizhu12/MccSTN
cd MccSTN
```
or other packages that are reminded to be installed.

#### Models
**Test:**
- Download the pre-trained [vgg_normalised.pth](https://drive.google.com/file/d/1PUXro9eqHpPs_JwmVe47xY692N3-G9MD/view?usp=sharing), place it at path `models/`.

- Download pre-trained models from this [MccSTN](https://pan.baidu.com/s/135zSFIU6EQSh1ohdAdSltw?pwd=zbi4).extraction code：zbi4 .Unzip and place them at path

`python test.py --content inputs/content/1.jpg --style inputs/style/1.jpg`

- Test two collections of images:
  `python test.py --content_dir inputs/content/ --style_dir inputs/style/`
![qualitative](https://github.com/haizhu12/MccSTN/assets/93024130/e92d2976-1cc7-4a50-96c6-c8aa58b8f5f1)

  
**train:**
Upcoming Releases
### Runtime Controls:

**Content-style trade-off:**
`python test.py --content inputs/content/*.jpg --style inputs/style/*.jpg --alpha 0.5`
![stylcontrol](https://github.com/haizhu12/MccSTN/assets/93024130/de1bee9f-4559-42c5-8e23-3ee18fa4e081)







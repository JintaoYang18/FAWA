# FAWA: Fast Adversarial Watermark Attack
Hao Jiang; Jintao Yang; Guang Hua*; Lixia Li; Ying Wang; Shenghui Tu; Song Xia

[*: corresponding author]


This repository contains the implementation of our paper [**FAWA: Fast Adversarial Watermark Attack**](https://ieeexplore.ieee.org/document/9376658)


***

If you find this code or the paper useful, please consider citing:

```
@ARTICLE{fawa,
  author={Jiang, Hao and Yang, Jintao and Hua, Guang and Li, Lixia and Wang, Ying and Tu, Shenghui and Xia, Song},
  journal={IEEE Transactions on Computers}, 
  title={FAWA: Fast Adversarial Watermark Attack}, 
  year={2021},
  volume={},
  number={},
  pages={1-13},
  doi={10.1109/TC.2021.3065172}}
```

#### And if you do anything interesting with this code we'd be happy to hear from you what it was.


## Contents


1. [Installation](#installation)
2. [Usage](#usage)
    - [Generate fawa examples](#generate-fawa-examples)
    - [Evaluate fawa examples](#evaluate-fawa-examples)
    - [Visualization (CAM)](#visualization-cam)


## Installation


1. Clone this repository:
```
git clone https://github.com/JintaoYang18/FAWA
cd FAWA/
```

2. Install conda envs and requirements:
```
conda env create -f fawa_env.yaml
pip install -r fawa_requirements.txt
```

  Note: if you don't have a GPU, install the cpu version of PyTorch. (We have not tested this setting.)


3. Prepare your dataset and put it into `100_image_class_950_999_300resize` directory.


4. Modify the 3 .txt files in the root directory according to your own data.

Note: We explained the role of each .txt file in detail in the `main.py` file.


5. Create `pre_trained_models` directories and download pre-trained `.pth` files:
```
mkdir pre_trained_models
cd pre_trained_models/
```
Download [**vgg-16 pre-trained .pth file**](https://download.pytorch.org/models/vgg16-397923af.pth), and put it in the `pre_trained_models` directory.



## Usage

### Generate fawa examples
```
python main.py
```

Note: The time it takes to generate the image depends on the performance of the computer, and you may have to wait.
The running time can be adjusted by modifying `p_size` and `g_round`.
You can even reduce the dimension of the problem, such as removing the rotation item.


### Evaluate fawa examples

Note: Use `your own model` or `pre-trained model` to evaluate fawa adversarial examples.


### Visualization (CAM)

Note:  You can use open source code, such as:
* https://github.com/utkuozbulak/pytorch-cnn-visualizations
+ https://github.com/jacobgil/pytorch-grad-cam

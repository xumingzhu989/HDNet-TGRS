# HDNet: A Hybrid Domain Network with Multi-Scale High-Frequency Information Enhancement for Infrared Small Target Detection

Mingzhu Xu, Chenglong Yu, Zexuan Li, Haoyu Tang, Yupeng Hu, Liqiang Nie, IEEE Transactions on Geoscience and Remote Sensing 2025.

## Structure
![](Fig/Structure.png)

## Introduction
This repository is the official implementation of our TGRS 2025 paper: [HDNet: A Hybrid Domain Network with Multi-Scale High-Frequency Information Enhancement for Infrared Small Target Detection](https://ieeexplore.ieee.org/document/11017756).

In this paper, we propose a novel Hybrid Domain Network (HDNet), which fuses frequency-domain features with conventional spatial-domain CNN features to markedly enhance target-background contrast and explicitly suppress background interference. Specifically, HDNet comprises two main branches: the spatial domain branch and the frequency domain branch. In the spatial domain, we innovatively introduce a Multi-scale Atrous Contrast convolution (MAC) module, utilizing multiple parallel atrous contrast convolutions with varying kernel sizes to enhance perception of small, variably sized targets. In the frequency domain, we have specifically designed the Dynamic High-Pass Filter (DHPF) module, hierarchically calculating low-frequency signal energy and dynamically removing specific low-frequency information to preserve high-frequency image details. This effectively filters out slowly varying low frequency backgrounds, highlighting small targets. Comprehensive ablation studies and experimental analysis on three datasets (IRSTD-1K, NUAA-SIRST, NUDT-SIRST) validate HDNet’s effectiveness and superiority compared to 26 state-of-the-art (SOTA) methods. The contribution of this paper are as follows:

1. We propose a novel Hybrid-Domain Network (HDNet) for IRSTD task. It takes advantage of the multi-scale target perception capability in the spatial domain and the low-frequency information suppression ability in the frequency domain to enhance the performance of IRSTD.
   
2. We propose a novel Multi-scale Atrous Contrast convolution (MAC) module in spatial domain. This module improves the contrast between targets and cluttered backgrounds, enhancing the perception capability of small and variable-sized targets.
   
3. We propose a novel Dynamic High-Pass Filter (DHPF) module in frequency domain. This module calculates the energy of low-frequency and dynamically removes a specific proportion of it, effectively suppressing the slowly varying low-frequency background interference.

4. We have conducted comprehensive ablation studies and experimental analysis on three public datasets (including IRSTD-1K, NUAA-SIRST, and NUDT-SIRST), validating the effectiveness and superiority of our HDNet, compared with 26 state-of-the-art (SOTA) methods.

## Datasets
Download the datasets and put them to './datasets': [IRSTD-1k](https://github.com/RuiZhang97/ISNet), [NUAA-SIRST](https://github.com/YimianDai/sirst), [NUDT-SIRST](https://github.com/YeRen123455/Infrared-Small-Target-Detection). Or test according to the train/val split ratio provided in the datasets directory.

## Prerequisite
Trained and tested on Ubuntu 22.04, with Python 3.10, PyTorch 2.1.0, Torchvision 0.16.2+cu121, CUDA 12.1, and 1×NVIDIA 3090.

## Training
The training command is very simple like this:
```
python main --dataset-dir --batch-size --epochs --mode 'train'
```

For example:
```
python main.py --dataset-dir './dataset/IRSTD-1k' --batch-size 4 --epochs 800 --mode 'train'
```

## Testing
You can test the model with the following command:
```
python main.py --dataset-dir './dataset/IRSTD-1k' --batch-size 4 --mode 'test' --weight-path './weight/irstd.pkl'
```

## Quantative Results
| Dataset    | mIoU (x10(-2)) | Pd (x10(-2)) | Fa (x10(-6)) |                                               Weights                                               |
| ---------- | :------------: | :----------: | :----------: | :-------------------------------------------------------------------------------------------------: |
| IRSTD-1k   |     70.26      |    94.56     |     4.33     |  [IRSTD-1k](https://drive.google.com/file/d/1WjKkkfIRlI7aNlu4xTglmVxwtDqlu4Gu/view?usp=drive_link)  |
| NUAA-SIRST |     79.17      |     100      |     0.53     | [NUAA-SIRST](https://drive.google.com/file/d/1GoCaiAEodUop5EPyDWu5LEfJ71D1kOz2/view?usp=drive_link) |
| NUDT-SIRST |     85.17      |    98.52     |     2.78     | [NUDT-SIRST](https://drive.google.com/file/d/1we0dE2L47z509-EW4_Bj4Y828oPSkNAe/view?usp=drive_link) |

## Qualitative Results
The detection results of HDNet on the three datasets: [HDNet_Visual_Result](https://drive.google.com/drive/folders/1RfoxhoHpjfbRMZHBOvISrJSB5lpoz40t?usp=drive_link). 

##
* HDNet employs the SLS loss and further improves the network architecture based on [MSHNet](https://github.com/Lliu666/MSHNet). Thanks to Qiankun Liu.

## Citation
**Please kindly cite the papers if this code is useful and helpful for your research.**

    @ARTICLE{11017756,
     author={Xu, Mingzhu and Yu, Chenglong and Li, Zexuan and Tang, Haoyu and Hu, Yupeng and Nie, Liqiang},
     journal={IEEE Transactions on Geoscience and Remote Sensing}, 
     title={HDNet: A Hybrid Domain Network With Multiscale High-Frequency Information Enhancement for Infrared Small-Target Detection}, 
     year={2025},
     volume={63},
     number={},
     pages={1-15},
     keywords={Frequency-domain analysis;Feature extraction;Object detection;Information filters;Convolution;Interference;Representation learning;Low-pass filters;Fast Fourier transforms;Background noise;Dynamic high-pass filter (DHPF);high-frequency information    enhancement;infrared small-target detection (IRSTD);multiscale atrous contrast (MAC) convolution},
     doi={10.1109/TGRS.2025.3574962}
     }


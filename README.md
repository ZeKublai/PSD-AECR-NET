# PSD-AECR-NET
 
Welcome! This repository contains my output for my NAPI 2022 internship where I combined PSD & AECR-Net for dehazing foggy images. See the files in the docs folder for details. Have fun experimenting!

# AECR-Net

Contrastive Learning for Compact Single Image Dehazing, CVPR2021. Official Pytorch based implementation.
<br>
[arxiv](https://arxiv.org/abs/2104.09367)
<br><br>
Please share them some love on their [GitHub](https://github.com/GlassyWu/AECR-Net)

# Principled S2R Dehazing
[**PSD: Principled Synthetic to Real Dehazing Guided by Physical Priors**](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_PSD_Principled_Synthetic-to-Real_Dehazing_Guided_by_Physical_Priors_CVPR_2021_paper.pdf)
<br>
[Zeyuan Chen](https://zychen-ustc.github.io/), Yangchao Wang, [Yang Yang](https://cfm.uestc.edu.cn/~yangyang/), [Dong Liu](http://staff.ustc.edu.cn/~dongeliu/)
<br>
CVPR 2021 (Oral)
<br><br>
Please share them some love on their [GitHub](https://github.com/zychen-ustc/PSD-Principled-Synthetic-to-Real-Dehazing-Guided-by-Physical-Priors)

### Environment
- Python 3.9.9
- Pytorch 1.10.1+cu113

## Downloads
Below are download links to the datasets, image outputs, pre-trained models respectively since they can't fit in this repository. Simply paste the folder into the same directory as the code and it should work as intended.
<br>
Folder Name|File size|Download
:-:|:-:|:-:
Images|20GB|[Google Drive](https://drive.google.com/drive/folders/1bYIErQICTjfKrdFU6usmfrlL-YO_v4Ac?usp=sharing)
Outputs|812MB|[Google Drive](https://drive.google.com/drive/folders/1eKM09TdwzM-y93TzJKx6m4fH82bIcMJr?usp=sharing)
Pre-trained Models|567MB|[Google Drive](https://drive.google.com/drive/folders/1yld8_NVdDNE64Tb0i4IALg07cRubGp61?usp=sharing)

## Notebooks
Notebook|Description
:-:|:-:
AECRNet_train.ipynb|Contains training code for AECR-Net using OTS dataset
PSD CVPR Testbench.ipynb|Compares all of the model's image results using CVPR dataset
PSD SOTS Testbench.ipynb|Compares all of the model's image results using SOTS dataset
PSD Validate.ipynb|Validates all fo the models using SOTS dataset
UNET_train.ipynb|Contains training code for U-Net using OTS dataset
napi_student_2021.ipynb|Initial practice exercise containing cropping & resizing code of OTS dataset

## Testing 
Runs the CVPR dataset to the PSD AECR-Net model and saves the output images to output/AERCNET/
```
python test_aecr.py
```
Runs the CVPR dataset to the PSD FFA-Net model and saves the output images to output/FFA/
```
python test_ffa.py
```
Runs the CVPR dataset to the PSD GCA-Net model and saves the output images to output/GCA/
```
python test_gca.py
```
Runs the CVPR dataset to the PSD MSBDN-Net model and saves the output images to output/MSBDN/
```
python test_msbdn.py
```
Trains the PSD AECR-Net model to with the OTS dataset and outputs the trained model at output/
```
python train_aecr.py
```
- To run tests, I highly recommend running command prompt as administror since some modules won't work without admin privileges. You can change settings such as the number of images to output or training values by editing the code.


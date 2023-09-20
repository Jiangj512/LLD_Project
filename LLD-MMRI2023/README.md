# This folder stores the training and testing sets required for code training and testing

## Data directory structure：
```
├── LLD-MMRI2023
    ├── data
        ├── train
            ├── MR745_0_6.npy
            ├── MR745_1_6.npy
            ├── MR745_2_6.npy
            ···
    ├── test
        ├── C+A
            ├── MR1872_-1.nii.gz
            ···
        ├── C+Delay
            ├── MR1872_-1.nii.gz
            ···
        ├── C+V
            ├── MR1872_-1.nii.gz
            ···
        ├── C-pre
            ├── MR1872_-1.nii.gz
            ···
        ├── DWI
            ├── MR1872_-1.nii.gz
            ···
        ├── In Phase
            ├── MR1872_-1.nii.gz
            ···
        ├── Out Phase
            ├── MR1872_-1.nii.gz
            ···
        ├── T2WI
            ├── MR1872_-1.nii.gz
            ···
```
Descriptions:

**data/train**: There are train and validation data in the ```data/train``` directory. Each file is a single slice stitched together from slices in the same position in each of the eight modalities of a patient. There are a number of slices for each patient, and the slice name consists of image_id, slice number, and label. 

**test**: There are test data in the ```test``` directory. Eight folders contain eight modal images for each patient. The filename consists of the image_id and the label.

## Download: 
**Due to space limitations, please visit our web site [MediSegLearner](https://pan.baidu.com/s/1UFbIR2PZJh4Fxb2DnUOLHA?pwd=n31z) to download the complete data separately(path: MediSegLearner/LLD-MMRI2023).**

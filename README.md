# Attention-cnn-MS-segmentation
An attention Fully Convolutional Densenet, based on ... , for Multiple Sclerosis lesion segmentation from FLAIR MRI

[https://arxiv.org/pdf/2304.10790.pdf](https://arxiv.org/pdf/2304.10790.pdf)

Our framework achieves more accuracy in multiple sclerosis lesion segmentation tasks compared with other 2D/3D segmentation methods.

## Dataset 
We release the codes which support the training and testing process of ISBI2015.

ISBI2015 download: https://smart-stats-tools.org/lesion-challenge-2015

Once you have downloaded the dataset, you must convert the nifti files into images and the folders must follow the following structure:
```bash
  dataset
└───ISBI_2015
    ├───test1
    │   │───Patient1
    ├───test1annot
    │   │───Patient1
    ├───...
    ├───train1
    │   │───P1_T1
    │   │───...
    │   │───...
    │   └───P5_T4
    ├───train1annot
    │   │───P1_T1
    │   │───...
    │   │───...
    │   └───P5_T4
    ├───...
    ├───train5
    │   │───P1_T1
    │   │───...
    │   │───...
    │   └───P5_T4
    ├───train5annot
    │   │───P1_T1
    │   │───...
    │   │───...
    │   └───P5_T4
    ├───val1
    │   │───P1_T1
    │   │───...
    │   │───...
    │   └───P5_T4
    ├───val1annot
    │   │───P1_T1
    │   │───...
    │   │───...
    │   └───P5_T4
    ├───...
    ├───val5
    │   │───P1_T1
    │   │───...
    │   │───...
    │   └───P5_T4
    └───val5annot
        │───P1_T1
        │───...
        │───...
        └───P5_T4

```
N.B. Note that the folders named train1-5 and val1-5 contain the folders of different patients (Fold), while the folders named train1-5annot and val1-5annot contain the respective masks.

Once the data has been downloaded and arranged according to that directory tree, the training process can begin.

## Training 

Use this command for training. You can also change the training parameters to change directories and exclude/include parts of the model (refer to train.py).

```bash
python train.py
```

## Testing

When you have trained a model, please modify the model path (--weights-fname), then run the code. You can also change the testing parameters  (refer to test.py).
```bash
python test.py
```

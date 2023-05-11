# attention-cnn-MS-segmentation
An attention Fully Convolutional Densenet, based on ... , for Multiple Sclerosis lesion segmentation from FLAIR MRI

[https://arxiv.org/pdf/2304.10790.pdf](https://arxiv.org/pdf/2304.10790.pdf)

Our framework achieves more accuracy in multiple sclerosis lesion segmentation tasks compared with other 2D/3D segmentation methods.

## dataset 
We release the codes which support the training and testing process of ISBI2015.

ISBI2015 download: https://smart-stats-tools.org/lesion-challenge-2015

Once you have downloaded the dataset, you must convert the nifti files into images and the folders must follow the following structure:
dataset/
  ISBI_2015/
    train1/
    ...
    train5/
    train1annot/
    ...
    train5annot/
    val1/
    ...
    val5/
    val1annot/
    ...
    val5annot/
    

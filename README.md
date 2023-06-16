# Boosting multiple sclerosis lesion segmentation through attention mechanism
An attention-based Fully Convolutional Densenet for Multiple Sclerosis lesion segmentation from FLAIR MRI

This repository accompanies the paper: [Boosting multiple sclerosis lesion segmentation through attention mechanism](https://www.sciencedirect.com/science/article/pii/S0010482523004869).

If you are using this software, please use the below BibTeX entry to cite our work:

```
@article{rondinella2023boosting,
  title={Boosting multiple sclerosis lesion segmentation through attention mechanism},
  author={Rondinella, Alessia and Crispino, Elena and Guarnera, Francesco and Giudice, Oliver and Ortis, Alessandro and Russo, Giulia and Di Lorenzo, Clara and Maimone, Davide and Pappalardo, Francesco and Battiato, Sebastiano},
  journal={Computers in Biology and Medicine},
  volume={161},
  pages={107021},
  year={2023},
  publisher={Elsevier}
}
```

This program is free software: it is distributed WITHOUT ANY WARRANTY.

## Installation

Create a conda or miniconda environment with the following commands:

```bash
conda create -n boosting-ms python=3.9
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install matplotlib scipy
```

## Dataset 
We release the codes which support the training and testing process on the ISBI2015 dataset.

The original ISBI2015 dataset can be downloaded here: [https://smart-stats-tools.org/lesion-challenge-2015](https://smart-stats-tools.org/lesion-challenge-2015)

At first, data must be converted: the nifti files into images and the folders must follow the following structure:

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
Please note that the folders named train1-5 and val1-5 contain the folders of different patients (Fold), while the folders named train1-5annot and val1-5annot contain the respective masks. Once the data has been downloaded and arranged according to that directory tree, the training process can begin.

For better reproducibility the ISBI dataset, already pre-processed and organized, ready for the training step, can be downloaded here: [ISBI dataset ready for training download link](https://iplab.dmi.unict.it/mfs/dataset/alessiarondinella/ISBI_2015.tar)

## Training 

Use this command for training. You can also change the training parameters to change directories and exclude/include parts of the model (refer to train.py).

```bash
conda activate boosting-ms
python train.py
```

## Testing

When training is finished, please modify the model path accordingly (--weights-fname), then run the test.py. You can also change the testing parameters  (refer to test.py).
```bash
conda activate boosting-ms
python test.py
```

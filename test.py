import torch
import torchvision.transforms as transforms
from models import tiramisu
from datasets import MSDataset
from datasets import joint_transforms
import utils.training as train_utils
from pathlib import Path
import os
import numpy
import random


# Define options
opt_defs = {}

# Dataset options
opt_defs["n_classes"] = dict(flags = ('-nc', '--nclasses'), info=dict(default=2, type=int, help="num of classes"))
opt_defs["mean"] = dict(flags = ('-mean', '--mean'), info=dict(default=0.1026, type=float, help="mean for dataset normalization"))
opt_defs["std"] = dict(flags = ('-std', '--std'), info=dict(default=0.0971, type=float, help="std for dataset normalization"))
#opt_defs["dataset_path"] = dict(flags = ('-dp', '--dataset-path'), info=dict(default="../../dataset/MS_Scan/dataset_Test_P2_T4", type=str, help="path to dataset on IPLAB"))
#opt_defs["dataset_path"] = dict(flags = ('-dp', '--dataset-path'), info=dict(default="./dataset/ISBI_2015", type=str, help="path to dataset"))
opt_defs["dataset_path"] = dict(flags = ('-dp', '--dataset-path'), info=dict(default="D:/Alessia/2_MS_Project_Gruppo_Imaging/dataset/ISBI_2015/ISBI_2015_PC", type=str, help="path to dataset on PC"))
opt_defs["test_dataset"] = dict(flags = ('-td','--test-dataset',), info=dict(default='test1', type=str, help="test1-5"))
opt_defs["weights_path"] = dict(flags = ('-wp', '--weights-path'), info=dict(default="./tiramisu_weights_ms/", type=str, help="path to weights"))
opt_defs["base_output_path"] = dict(flags = ('-bop', '--base-output-path'), info=dict(default="/Patient-", type=str, help="where to save output"))
opt_defs["patient_name"] = dict(flags = ('-pn', '--patient-name'), info=dict(default='Patient-', type=str, help="patient name"))
opt_defs["patients"] = dict(flags = ('-patients', '--patients'), info=dict(default=[1], nargs ='+', type=int, help="patients to test"))

# Model options
opt_defs["lstm_kernel_size"] = dict(flags = ('-lstmkernel','--lstm-kernel-size',), info=dict(default=3, type=int, help="lstm kernel size"))
opt_defs["lstm_num_layers"] = dict(flags = ('-lstmnumlayers','--lstm-num-layers',), info=dict(default=1, type=int, help="lstm kernel size"))
opt_defs["use_sa"] = dict(flags = ('-usesa', '--use-sa'), info=dict(default=True, type=bool, help="use Squeeze and Attention blocks (use:True, not use: False)"))
opt_defs["use_stn"] = dict(flags = ('-usestn', '--use-stn'), info=dict(default=False,  type=bool, help="use stn (use:True, not use: False)"))
opt_defs["use_lstm"] = dict(flags = ('-ulstm', '--use-lstm'), info=dict(default=False, type=bool, help="use lstm (use:True, not use: False)"))
opt_defs["seq_size"] = dict(flags = ('-ss', '--seq-size'), info=dict(default=1, type=int, help="sequence size"))
opt_defs["sliding_window"] = dict(flags = ('-sw', '--sliding-window'), info=dict(default=True, type=bool, help="use sliding window (compute the loss only w.r.t. the central slice in the sequence)(use:True, not use: False)"))
opt_defs["bidirectional"] = dict(flags = ('-bi', '--bidirectional'), info=dict(default=False, type=bool, help="bidirectional c-lstm (use:True, not use: False)"))
opt_defs["input_dim"] = dict(flags = ('-dim', '--input-dim'), info=dict(default=160, type=int, help="input dim"))

# Training options
opt_defs["optim"] = dict(flags = ('-o', '--optim'), info=dict(default="RMSprop", help="optimizer"))
opt_defs["learning_rate"] = dict(flags = ('-lr', '--learning-rate'), info=dict(default=1e-4, type=float, help="learning rate"))
opt_defs["learning_rate_decay_by"] = dict(flags = ('-lrdb', '--learning-rate-decay-by'), info=dict(default=0.995, type=float, help="learning rate decay factor"))
opt_defs["learning_rate_decay_every"] = dict(flags = ('-lrde', '--learning-rate-decay-every'), info=dict(default=10, type=int, help="learning rate decay period"))
opt_defs["weight_decay"] = dict(flags = ('-wd', '--weight-decay',), info=dict(default=1e-4, type=float, help="weight decay"))
opt_defs["loss_type"] = dict(flags = ('-lt', '--loss-type'), info=dict(default='dice', type = str, help="the type of loss, i.e. dice"))

# Checkpoint options
opt_defs["weights_fname"] = dict(flags = ('-wf', '--weights-fname'), info=dict(default=None, type=str, help="weights file name, i.e. 'weights-#tag-#folder-#epochs.pth'"))

# Read options
import argparse
parser = argparse.ArgumentParser()
for k,arg in opt_defs.items():
    print(arg["flags"])
    parser.add_argument(*arg["flags"], **arg["info"])
opt = parser.parse_args(None)
print(opt)


#Dataset option
n_classes = opt.nclasses
mean = opt.mean
std = opt.std
DATASET_PATH = Path(opt.dataset_path)
test_dir = opt.test_dataset
WEIGHTS_PATH = Path(opt.weights_path)
base_output_path = opt.base_output_path
patient_name = opt.patient_name
patients = opt.patients

# Model options
lstm_kernel_size = opt.lstm_kernel_size
lstm_num_layers = opt.lstm_num_layers
use_sa = opt.use_sa
use_stn = opt.use_stn
use_lstm = opt.use_lstm
seq_size = opt.seq_size
sliding_window = opt.sliding_window
bidirectional = opt.bidirectional
input_dim = opt.input_dim

# Training options
optim = opt.optim
lr = opt.learning_rate
lr_decay = opt.learning_rate_decay_by
weight_decay = opt.weight_decay
loss_type = opt.loss_type
num_epochs = opt.num_epochs

# Checkpoint options
weights_fname = opt.weights_fname


torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic=True
seed = 1
random.seed(seed)
numpy.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)


if seq_size > 1:
    use_lstm = True
    use_se = True
else:
    use_lstm = False
    use_se = True

#SET DEVICE
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(dev)

dice_vector = []
sens_vector = []
spec_vector = []

if __name__ == '__main__':

    model = tiramisu.FCDenseNet67(loss_type=loss_type, n_classes=2, grow_rate = 12, use_stn=use_stn, use_se=use_se, seq_size=seq_size, use_lstm=use_lstm,
                                lstm_kernel_size=lstm_kernel_size, lstm_num_layers=lstm_num_layers,
                                bidirectional=bidirectional)
    model = model.to(dev)

    #SET OPTIMIZER
    if optim == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError("Optimizer chosen not implemented!")

    train_utils.load_weights(model, optimizer, os.path.join(WEIGHTS_PATH, weights_fname))
    model.eval()
    for p in patients:
        print(p)

        test_dset = MSDataset.MSDataset(
        DATASET_PATH, test_dir, joint_transform=None,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),seq_size=seq_size, sliding_window = sliding_window, input_dim = input_dim, patient_name = patient_name + str(p))

        test_loader = torch.utils.data.DataLoader(
        test_dset, batch_size=1, shuffle=False)
        print("Test patient: %d, size %d" % (p, len(test_loader.dataset.imgs)))

        #set base_output_path
        weights_fname_ = weights_fname.split('.pth')
        weights_fname__ = weights_fname_.split('-')
        fold_id = weights_fname__[-2]
        print("Fold" + fold_id + base_output_path + str(p))
        OUTPUT_PATH = Path("Fold" + fold_id + base_output_path + str(p))
        OUTPUT_PATH.mkdir(exist_ok=True)

        test_dice, test_sens, test_spec, test_acc, test_err, test_ppv, test_npv, test_extra_frac, text_iou = train_utils.compute_output(model, test_loader, OUTPUT_PATH, seq_size, sliding_window)
    
        print("Dice: %4f" % test_dice)
        dice_vector.append(test_dice)
        print("Sens: %4f" % test_sens)
        sens_vector.append(test_sens)
        print("Spec: %4f" % test_spec)
        spec_vector.append(test_spec)
        print("Acc: %4f" % test_acc)
        print("Err: %4f" % test_err)
        print("PPV: %4f" % test_ppv)
        print("NPV: %4f" % test_npv)
        print("Extra Fraction: %4f" % test_extra_frac)
        print("IOU: %4f" % text_iou)


    print("Dice MAX: %4f" % numpy.max(numpy.array(dice_vector)))
    print("Dice mean: %4f" % numpy.mean(numpy.array(dice_vector)))
    print("Dice std: %4f" % numpy.std(numpy.array(dice_vector)))

    print("Sens mean: %4f" % numpy.mean(numpy.array(sens_vector)))
    print("Sens std: %4f" % numpy.std(numpy.array(sens_vector)))

    print("Spec mean: %4f" % numpy.mean(numpy.array(spec_vector)))
    print("Spec std: %4f" % numpy.std(numpy.array(spec_vector)))

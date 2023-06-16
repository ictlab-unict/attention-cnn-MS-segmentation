from pathlib import Path
import numpy
import random
import os
import time
import torch
import torchvision.transforms as transforms
from models import tiramisu
from datasets import MSDataset
from datasets import joint_transforms
import utils.training as train_utils
import gc
import matplotlib.pyplot as plt


# Define options
opt_defs = {}

# Dataset options
opt_defs["n_classes"] = dict(flags = ('-nc', '--nclasses'), info=dict(default=2, type=int, help="num of classes"))
opt_defs["mean"] = dict(flags = ('-mean', '--mean'), info=dict(default=0.1026, type=float, help="mean for dataset normalization"))
opt_defs["std"] = dict(flags = ('-std', '--std'), info=dict(default=0.0971, type=float, help="std for dataset normalization"))
opt_defs["dataset_path"] = dict(flags = ('-dp', '--dataset-path'), info=dict(default="./datasets/ISBI_2015", type=str, help="path to dataset"))
opt_defs["validation_dataset"] = dict(flags = ('-vd','--val-dataset',), info=dict(default='val', type=str, help="val or test"))
opt_defs["folders"] = dict(flags = ('-f','--folders',), info=dict(default=5, type=int, help="num folders for cross validation"))

# Model options
opt_defs["lstm_kernel_size"] = dict(flags = ('-lstmkernel','--lstm-kernel-size',), info=dict(default=3, type=int, help="lstm kernel size"))
opt_defs["lstm_num_layers"] = dict(flags = ('-lstmnumlayers','--lstm-num-layers',), info=dict(default=1, type=int, help="lstm kernel size"))
opt_defs["use_sa"] = dict(flags = ('-usesa', '--use-sa'), info=dict(default=True,  type=bool, help="use Squeeze and Attention blocks (use:True, not use: False)"))
opt_defs["use_stn"] = dict(flags = ('-usestn', '--use-stn'), info=dict(default=False,  type=bool, help="use stn (use:True, not use: False)"))
opt_defs["use_lstm"] = dict(flags = ('-uselstm', '--use-lstm'), info=dict(default=True,  type=bool, help="use lstm layers (use:True, not use: False)"))
opt_defs["seq_size"] = dict(flags = ('-ss', '--seq-size'), info=dict(default=3, type=int, help="sequence size to test"))
opt_defs["sliding_window"] = dict(flags = ('-swv', '--sliding-window'), info=dict(default=True,  type=bool, help="sliding window (compute the loss only w.r.t. the central slice in the sequence) to test (use:True, not use: False)"))
opt_defs["bidirectional"] = dict(flags = ('-bv', '--bidirectional'), info=dict(default=False,  type=bool, help="bidirectional c-lstm to test (use:True, not use: False)"))
opt_defs["input_dim"] = dict(flags = ('-dim', '--input-dim'), info=dict(default=160, type=int, help="input dim")) #224

# Training options
opt_defs["batch_size"] = dict(flags = ('-b', '--batch-size'), info=dict(default=1, type=int, help="batch size"))
opt_defs["optim"] = dict(flags = ('-o', '--optim'), info=dict(default="RMSprop", help="optimizer"))
opt_defs["learning_rate"] = dict(flags = ('-lr', '--learning-rate'), info=dict(default=1e-4, type=float, help="learning rate"))
opt_defs["learning_rate_decay_by"] = dict(flags = ('-lrdb', '--learning-rate-decay-by'), info=dict(default=0.995, type=float, help="learning rate decay factor"))
opt_defs["learning_rate_decay_every"] = dict(flags = ('-lrde', '--learning-rate-decay-every'), info=dict(default=10, type=int, help="learning rate decay period"))
opt_defs["weight_decay"] = dict(flags = ('-wd', '--weight-decay',), info=dict(default=1e-4, type=float, help="weight decay"))
opt_defs["loss_type"] = dict(flags = ('-lt', '--loss-type'), info=dict(default='dice', type = str, help="the type of loss, i.e. dice"))
opt_defs["num_epochs"] = dict(flags = ('-ne', '--num-epochs',), info=dict(default=2, type=int, help="training epochs"))

# Checkpoint options
opt_defs["results_path"] = dict(flags = ('-rp', '--results-path'), info=dict(default="./results_ms/", type=str, help="path to results"))
opt_defs["weights_path"] = dict(flags = ('-wp', '--weights-path'), info=dict(default="./tiramisu_weights_ms/", type=str, help="path to weights"))
opt_defs["weights_fname"] = dict(flags = ('-wf', '--weights-fname'), info=dict(default=None, type=str, help="weights file name, i.e. 'weights-#tag-#folder-#epochs.pth'"))
opt_defs["last_tag"] = dict(flags = ('-t', '--last-tag'), info=dict(default=0, type=int, help="last checkpoint tag"))


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
val_dataset = opt.val_dataset
folders = opt.folders

#MODEL OPTIONS
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
batch_size = opt.batch_size
optim = opt.optim
lr = opt.learning_rate
lr_decay = opt.learning_rate_decay_by
decay_every_n_epochs = opt.learning_rate_decay_every
weight_decay = opt.weight_decay
loss_type = opt.loss_type
num_epochs = opt.num_epochs

# Checkpoint options
RESULTS_PATH = Path(opt.results_path)
RESULTS_PATH.mkdir(exist_ok=True)
WEIGHTS_PATH = Path(opt.weights_path)
WEIGHTS_PATH.mkdir(exist_ok=True)
weights_fname = opt.weights_fname
last_tag = opt.last_tag

seed = 1
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic=True

if seq_size > 1:
    use_lstm = True
    use_se = True
else:
    use_lstm = False
    use_se = True

#TAG TRAINING
tag = last_tag + 1
last_tag = tag


train_joint_transformer = transforms.Compose([
        joint_transforms.JointScale(input_dim),
        joint_transforms.JointCenterCrop(input_dim),
        #joint_transforms.JointRandomHorizontalFlip(),
        #joint_transforms.JointRandomRotation(90),
        #joint_transforms.JointRandomRotation(270),
        #joint_transforms.JointRandomAffine(0.2)
        ])

eval_joint_transformer = transforms.Compose([
    joint_transforms.JointScale(input_dim),
    joint_transforms.JointCenterCrop(input_dim)])

#SET DEVICE
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(dev)

print("TAG: ", tag)
if __name__ == '__main__':

    #CROSS-VALIDATION
    sum_dice = 0
    for f in range(1, folders + 1):
        if folders == 1:
            folder = ''
        else:
            folder = f
        print('FOLD:' + str(folder))
        train_dset = MSDataset.MSDataset(DATASET_PATH, 'train' + str(folder), input_dim = input_dim, mean = mean, std = std,
                                                        joint_transform=train_joint_transformer,
                                                        transform=transforms.Compose([transforms.Grayscale(3),
                                                            transforms.ToTensor(),
                                                        ]), seq_size=seq_size,
                                                        sliding_window=sliding_window)
                            
        print("TRAIN FOLD" + str(folder) + " LENGHT: ", len(train_dset))
        val_dset = MSDataset.MSDataset(
            DATASET_PATH, val_dataset + str(folder), input_dim = input_dim, mean = mean, std = std, joint_transform=eval_joint_transformer,
            transform=transforms.Compose([transforms.Grayscale(3),
                transforms.ToTensor(),
            ]), seq_size=seq_size, sliding_window=sliding_window)
        print("VAL FOLD" + str(folder) + " LENGHT: ", len(val_dset))

        random.seed(seed)
        numpy.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)

        train_loader = torch.utils.data.DataLoader(
                train_dset, batch_size=batch_size, shuffle=True, num_workers=0)

        val_loader = torch.utils.data.DataLoader(
                val_dset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        #SHOW SINGLE BATCH
        #train_utils.show_batch(val_loader)

        model = tiramisu.FCDenseNet67(loss_type = loss_type, n_classes=n_classes, grow_rate = 12, use_stn=use_stn, use_sa = use_sa, seq_size=seq_size, use_lstm=use_lstm,
                                        lstm_kernel_size=lstm_kernel_size, lstm_num_layers=lstm_num_layers,
                                        bidirectional=bidirectional)
        model = model.to(dev)


        #SHOW MODEL
        #print(model)

        #SET OPTIMIZER
        if optim == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optim == 'RMSprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError("Optimizer chosen not implemented!")

        #LOAD WEIGHTS
        if weights_fname is None:
            model.apply(train_utils.weights_init)
            start_epoch = 1
        else:
            start_epoch, history_loss_train, history_loss_val, history_acc_train, history_acc_val, history_DSC_val, history_sens_val, history_spec_val = train_utils.load_weights(model, optimizer, os.path.join(WEIGHTS_PATH, weights_fname))

        #VARIABLES
        history_loss_train=[]
        history_acc_train=[]
        history_loss_val=[]
        history_acc_val=[]
        history_sens_val=[]
        history_spec_val=[]
        history_ppv_val=[]
        history_npv_val=[]
        history_DSC_train =[]
        history_DSC_val = []

        criterion = None
        max_dice = 0
        epoch_max_dice = 1
        list_val_dice = []
        
        for epoch in range(start_epoch, num_epochs + start_epoch):
            since = time.time()

            ### Train ###
            trn_loss, trn_err, trn_dice, trn_sens, trn_spec = train_utils.train(
                model, train_loader, optimizer, criterion, seq_size, sliding_window, loss_type)
            print("-" * 150)
            print('Epoch {:d}\nTrain - Loss: {:.4f}, Acc: {:.4f}, Dice: {:.4f}, Sens: {:.4f}, Spec: {:.4f}'.format(
                epoch, trn_loss, 1 - trn_err, trn_dice, trn_sens, trn_spec))

            #plot
            history_loss_train.append(trn_loss)
            history_acc_train.append(1 - trn_err)
            history_DSC_train.append(trn_dice)

            time_elapsed = time.time() - since
            print('Train Time {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            gc.collect()

            ### Test ###
            val_loss, val_err, val_dice, val_sens, val_spec, val_ppv, val_npv = train_utils.test(model, val_loader, criterion, seq_size, sliding_window, loss_type)
            print('Val - Loss: {:.4f}, Acc: {:.4f}, Dice: {:.4f}, Sens: {:.4f}, Spec: {:.4f}'.format(val_loss, 1 - val_err, val_dice, val_sens, val_spec))
            
            #plot
            #per i plot
            history_loss_val.append(val_loss)
            history_acc_val.append(1 - val_err)
            history_sens_val.append(val_sens)
            history_spec_val.append(val_spec)
            history_ppv_val.append(val_ppv)
            history_npv_val.append(val_npv)

            time_elapsed = time.time() - since
            print('Total Time {:.0f}m {:.0f}s\n'.format(
                time_elapsed // 60, time_elapsed % 60))

            history_DSC_val.append(val_dice)
            list_val_dice.append(val_dice)

            if val_dice > max_dice:
                max_dice = val_dice
                epoch_max_dice = epoch
                train_utils.save_weights(model, optimizer, tag, f, epoch, val_loss, val_err, val_dice, history_loss_train, history_loss_val, history_acc_train, history_acc_val, history_DSC_val, history_sens_val, history_spec_val, WEIGHTS_PATH)
            
            ### Adjust Lr ###
            train_utils.adjust_learning_rate(lr, lr_decay, optimizer,
                                                epoch, decay_every_n_epochs)

        sum_dice = sum_dice + max_dice

        avg_dice = sum(list_val_dice) / len(list_val_dice)
        print('Max Dice Folder {:d}: {:.4f} at epoch {:d}'.format(f, max_dice, epoch_max_dice))
        print('Mean Dice Folder {:d}: {:.4f}'.format(f, avg_dice))

        # Plot loss history
        plt.title("Loss FCDenseNet67, Epoche: "+ str(len(history_loss_train))+", Batch_size: " + str(batch_size) + ", LR: " + str(lr))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(history_loss_train, label='train')
        plt.plot(history_loss_val, label='val')
        plt.legend()
        plt.savefig('results_ms/Loss_FCDenseNet67_numepoche_{}_bs{}_tag{}_folder{}.png'.format(num_epochs, batch_size, tag, f))
        plt.close()

        # Plot accuracy history
        plt.title("Accuracy FCDenseNet67, Epoche: "+ str(len(history_loss_train))+", Batch_size: " + str(batch_size) + ", LR: " + str(lr))
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.plot(history_acc_train, label='train')
        plt.plot(history_acc_val, label='val')
        plt.legend()
        plt.savefig('results_ms/Accuracy_FCDenseNet67_numepoche_{}_bs{}_tag{}_folder{}.png'.format(num_epochs, batch_size, tag, f))
        plt.close()

        # Plot DSC history
        plt.title("DSC FCDenseNet67, Epoche: "+ str(len(history_DSC_val))+", Batch_size: " + str(batch_size) + ", LR: " + str(lr))
        plt.xlabel('Epoch')
        plt.ylabel('DSC')
        plt.plot(history_DSC_train, label='train')
        plt.plot(history_DSC_val, label='val')
        plt.legend()
        plt.savefig('results_ms/DSC_FCDenseNet67_numepoche_{}_bs{}_tag{}_folder{}.png'.format(num_epochs, batch_size, tag, f))
        plt.close()
    
    mean_dice = sum_dice / folders
    print('Mean Max Dice Folders: {:.4f}', mean_dice)






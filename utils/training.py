import os
import sys
import math
import string
import random
import shutil
import time
import heapq

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from scipy import ndimage as ndi
import torch.nn.functional as F
import numpy as np
from operator import add
pixels_threshold = 10
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import binary_crossentropy
import keras.backend as k
import matplotlib.pyplot as plt

#from . import imgs as img_utils
#from utils import imgs as img_utils
import imageio
from pathlib import Path
import numpy
from torchvision.utils import make_grid

def show_batch(dl):
    for images, labels in dl:
        print(labels.size())
        print(images.size())
        for l in labels[0]:
            plt.imshow(l)
            plt.show()
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images[0][:900], nrow=10, normalize=True).permute(1, 2, 0))
        plt.show()
        break

def save_weights(model, optim, tag, epoch, loss, err, dice, history_loss_t, history_loss_v, history_accuracy_t, history_accuracy_v, history_DSC, history_sens_v, history_spec_v, weights_path):
    weights_fname = 'weights-%d-%d.pth' % (tag, epoch)
    weights_fpath = os.path.join(weights_path, weights_fname)
    torch.save({
        'startEpoch': epoch + 1,
        'loss': loss,
        'error': err,
        'dice': dice,
        'model_state': model.state_dict(),
        'optim_state': optim.state_dict(),
        'history_loss_t': history_loss_t,
        'history_loss_v': history_loss_v,
        'history_accuracy_t': history_accuracy_t,
        'history_accuracy_v': history_accuracy_v,
        'history_DSC': history_DSC,
        'history_sens_v': history_sens_v,
        'history_spec_v': history_spec_v
    }, weights_fpath)
    shutil.copyfile(weights_fpath, os.path.join(weights_path, 'latest.th'))

def load_weights(model, optimizer, fpath):
    print("loading weights '{}'".format(fpath))
    weights = torch.load(fpath)
    startEpoch = weights['startEpoch']
    history_loss_t = weights['history_loss_t']
    history_loss_v = weights['history_loss_v']
    history_accuracy_t = weights['history_accuracy_t']
    history_accuracy_v = weights['history_accuracy_v']
    history_DSC = weights['history_DSC']
    history_sens_v = weights['history_sens_v']
    history_spec_v = weights['history_spec_v']
    model.load_state_dict(weights['model_state'])
    optimizer.load_state_dict(weights['optim_state'])
    print("loaded weights (lastEpoch {}, loss {}, error {}, dice {})"
          .format(startEpoch - 1, weights['loss'], weights['error'], weights['dice']))
    return startEpoch, history_loss_t, history_loss_v, history_accuracy_t, history_accuracy_v, history_DSC, history_sens_v, history_spec_v



def adjust_learning_rate(lr, decay, optimizer, cur_epoch, n_epochs):
    """Sets the learning rate to the initially
        configured `lr` decayed by `decay` every `n_epochs`"""
    new_lr = lr * (decay ** (cur_epoch // n_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()

def get_predictions(output_batch):
    bs, c, h, w = output_batch.size()
    tensor = output_batch
    values, indices = tensor.cpu().max(1)
    indices = indices.view(bs, h, w)
    return indices

def error(preds, targets):
    assert preds.size() == targets.size()
    bs, h, w = preds.size()
    n_pixels = bs * h * w
    incorrect = preds.ne(targets).cpu().sum().item()
    err = incorrect / n_pixels
    return round(err, 5)

def dice_loss(outputs, target):
    smooth = 0.1

    outputs = outputs[:, 1, :, :]

    iflat = outputs.contiguous().view(-1)
    tflat = target.float().contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))

def dice(tp, fp, fn):
    if (2 * tp + fp + fn) > 0:
        dice = 2 * tp / (2 * tp + fp + fn)
        return round(dice, 5)
    else:
        return 1

def compute_performance(preds, targets):
    assert preds.size() == targets.size()
    tp = targets.mul(preds).eq(1).sum().item()
    fp = targets.eq(0).long().mul(preds).eq(1).sum().item()
    fn = preds.eq(0).long().mul(targets).eq(1).sum().item()
    tn = targets.eq(0).long().mul(preds).eq(0).sum().item()
    return tp, fp, fn, tn

def train(model, trn_loader, optimizer, criterion, seq_size, sliding_window, loss_type = 'dice'):
    model.train()
    trn_loss = 0
    trn_error = 0
    trn_tp = 0
    trn_fp = 0
    trn_fn = 0
    trn_tn = 0
    seq_window = (seq_size - 1) // 2
    for idx, data in enumerate(trn_loader):
        inputs = data[0].cuda()
        targets = data[1].cuda()
        targets = targets.view(targets.size(0) * targets.size(1), targets.size(2), targets.size(3))
        optimizer.zero_grad()
        outputs, deep_out = model(inputs)[:2]
        if sliding_window:
            indices = range(seq_window, outputs.size(0), seq_size)
            outputs = outputs[indices, :, :, :]
            targets = targets[indices, :, :]
        if loss_type == 'dice':
            loss = dice_loss(outputs, targets)
            
        else:
            loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        trn_loss += loss.item()
        preds = get_predictions(outputs)
        trn_error += error(preds, targets.cpu())
        tmp_tp, tmp_fp, tmp_fn, tmp_tn = compute_performance(preds, targets.cpu())
        trn_tp += tmp_tp
        trn_fp += tmp_fp
        trn_fn += tmp_fn
        trn_tn += tmp_tn

    trn_size = len(trn_loader)
    trn_loss /= trn_size
    trn_error /= trn_size
    trn_dice = dice(trn_tp, trn_fp, trn_fn)
    sens = trn_tp/(trn_tp + trn_fn)
    spec = trn_tn / (trn_tn + trn_fp)
    return trn_loss, trn_error, trn_dice, sens, spec


def test(model, test_loader, criterion, seq_size, sliding_window, loss_type = 'dice'):
    model.eval()
    test_loss = 0
    test_error = 0
    test_tp = 0
    test_fp = 0
    test_fn = 0
    test_tn = 0
    for inputs, targets in test_loader:
        with torch.no_grad():
            inputs = inputs.cuda()
            targets = targets.cuda()
            targets = targets.view(targets.size(0) * targets.size(1), targets.size(2), targets.size(3))
            outputs = model(inputs)[0]
            if sliding_window:
                seq_window = (seq_size - 1) // 2
                indices = range(seq_window, outputs.size(0), seq_size)
                outputs = outputs[indices, :, :, :]
                targets = targets[indices, :, :]

            if loss_type == 'dice':
                test_loss += dice_loss(outputs, targets).item()
            else:
                test_loss += criterion(outputs, targets).item()
            
            preds = get_predictions(outputs)
            test_error += error(preds, targets.cpu())
            tmp_tp, tmp_fp, tmp_fn, tmp_tn = compute_performance(preds, targets.cpu())
            dice(tmp_tp, tmp_fp, tmp_fn)
            test_tp += tmp_tp
            test_fp += tmp_fp
            test_fn += tmp_fn
            test_tn += tmp_tn

    test_size = len(test_loader)
    if test_size > 0:
        test_loss /= test_size
        test_error /= test_size
    test_dice = dice(test_tp, test_fp, test_fn)
    sens = test_tp / (test_tp + test_fn)
    spec = test_tn / (test_tn + test_fp)
    ppv = test_tp / (test_tp + test_fp)
    npv= test_tn / (test_tn + test_fn)
    return test_loss, test_error, test_dice, sens, spec, ppv, npv
import os
import shutil
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
from scipy import ndimage as ndi
from utils import imgs as img_utils
from torchvision.utils import save_image

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

def save_weights(model, optim, tag, folder, epoch, loss, err, dice, history_loss_t, history_loss_v, history_accuracy_t, history_accuracy_v, history_DSC, history_sens_v, history_spec_v, weights_path):
    weights_fname = 'weights-%d-%d-%d.pth' % (tag, folder, epoch)
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


#save slice + ground truth + prediction + FP + FN
def compute_output(model, test_loader, output_path, seq_size, sliding_window):
    if sliding_window:
        curr_seq_size = 1
        image_idx = seq_size // 2
    else:
        curr_seq_size = seq_size
        image_idx = 0
    test_tp = 0
    test_fp = 0
    test_fn = 0
    test_tn = 0
    dice_vector = []
    for inputs, targets in test_loader:
        with torch.no_grad():
            inputs = inputs.cuda()
            targets = targets.cuda()
            targets = targets.view(seq_size, targets.size(2), targets.size(3))
            outputs = model(inputs)[0]
            inputs = inputs.view(seq_size, inputs.size(2), inputs.size(3), inputs.size(4))

            if sliding_window:
                seq_window = (seq_size - 1) // 2
                indices = range(seq_window, outputs.size(0), seq_size)
                inputs = inputs[indices, :, :, :]
                outputs = outputs[indices, :, :, :]
                targets = targets[indices, :, :]
            pred = get_predictions(outputs)
            imgs_to_save = [] 
            for j in range(curr_seq_size):
                np_pred = pred[j]
                ejtema= np.zeros((160, 160))
                notditectedlesion=np.zeros((160, 160))
                wronglesiondetected=np.zeros((160, 160))
                b = targets[j] 
                for i in range(b.shape[0]-1):
                    for k in range(b.shape[1]-1):
                        
                        if b[i,k]==1 or np_pred[i,k]==1:
                            ejtema[i,k]=1
                            if b[i,k]==1 and np_pred[i,k]==0:
                                notditectedlesion[i,k]=1
                            if b[i,k]==0 and np_pred[i,k]==1:
                                wronglesiondetected[i,k]=1

                false_negative_mask = torch.from_numpy(ndi.binary_fill_holes(notditectedlesion).astype(int))
                false_positive_mask = torch.from_numpy(ndi.binary_fill_holes(wronglesiondetected).astype(int))

                imgs_to_save.append(img_utils.normalize(inputs[j].cpu()))
                t = targets[j].cpu().float().unsqueeze(0)
                
                imgs_to_save.append(torch.cat((t, t, t), 0))
                np_pred = np_pred.float().unsqueeze(0)
                imgs_to_save.append(torch.cat((np_pred, np_pred, np_pred), 0))

                false_negative_mask = false_negative_mask.float().unsqueeze(0)
                imgs_to_save.append(torch.cat((false_negative_mask, false_negative_mask, false_negative_mask), 0))

                false_positive_mask = false_positive_mask.float().unsqueeze(0)
                imgs_to_save.append(torch.cat((false_positive_mask, false_positive_mask, false_positive_mask), 0))

                img_fpath = os.path.join(output_path, str(image_idx) + '.png')

                save_image(imgs_to_save, img_fpath, nrow=5)
                
                #save single images
                output_path_img_separate = str(output_path) + '\separate_imgs'
                os.makedirs(output_path_img_separate,exist_ok= True)

                save_image(img_utils.normalize(inputs[j].cpu()), os.path.join(output_path_img_separate, str(image_idx) + '_slice.png'))
                save_image(torch.cat((t, t, t), 0), os.path.join(output_path_img_separate, str(image_idx) + '_ground_truth.png'))
                save_image(torch.cat((np_pred, np_pred, np_pred), 0), os.path.join(output_path_img_separate, str(image_idx) + '_pred.png'))
                save_image(torch.cat((false_negative_mask, false_negative_mask, false_negative_mask), 0), os.path.join(output_path_img_separate, str(image_idx) + '_FN.png'))
                save_image(torch.cat((false_positive_mask, false_positive_mask, false_positive_mask), 0), os.path.join(output_path_img_separate, str(image_idx) + '_FP.png'))
                
                image_idx = image_idx + 1
                
                tmp_tp, tmp_fp, tmp_fn, tmp_tn = compute_performance(pred[j], targets[j].cpu())

                test_tp += tmp_tp
                test_fp += tmp_fp
                test_fn += tmp_fn
                test_tn += tmp_tn           
                
    dsc = dice(test_tp, test_fp, test_fn)
    sens = test_tp / (test_tp + test_fn)
    spec = test_tn / (test_tn + test_fp)
    acc=(test_tp+test_tn)/(test_tp+test_tn+test_fn+test_fp)
    error=(test_fp+test_fn)/(test_tp+test_tn+test_fn+test_fp) 
    ppv = test_tp / (test_tp + test_fp)
    npv= test_tn / (test_tn + test_fn)
    extra_fraction=(test_fp)/(test_tn+test_fn)
    iou = test_tp / (test_tp + test_fn + test_fp)
   
    return dsc, sens, spec, acc, error, ppv, npv, extra_fraction, iou
    

import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
from torchvision.datasets.folder import default_loader


def _make_dataset(dir, patient_name = None):
    images = []
    for root, _, fnames in sorted(os.walk(os.path.join(dir + 'annot'))):
        if patient_name == None or root == os.path.join(dir + 'annot', patient_name):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = path
                im = Image.open(item)
                im=np.asanyarray(im)
                im = im.sum()
                if im>0: 
                    images.append(item)
    return images

class LabelToLongTensor(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            label = torch.from_numpy(pic).long()
        else:
            label = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes())) #label size[49152]
            label = label.view(pic.size[1], pic.size[0], 3) #label size [128, 128, 3]
            label = label[:,:,0].unsqueeze(2) #label size [128, 128, 1]          
            m = label.max() #255
            if m > 0:
                label = label/m
            label = label.transpose(0, 1).transpose(0, 2).squeeze().contiguous().float()#.long() #label size [128, 128]
            label = (label > 0.2).float() * 1
        return label

class MSDataset(data.Dataset):
    def __init__(self, root, split='train', input_dim = 128, mean = 0.1026, std = 0.0971, joint_transform=None,
                 transform=None, target_transform= LabelToLongTensor(),
                 download=False,
                 loader=default_loader, seq_size = 1, sliding_window = False, size = None, patient_name = None ):
        self.root = root
        self.input_dim = input_dim
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform
        self.loader = loader
        self.mean = mean
        self.std = std
        self.seq_size = seq_size
        self.sliding_window = sliding_window 
        self.imgs = _make_dataset(os.path.join(self.root, self.split), patient_name)

    def __len__(self):
        if self.sliding_window:
            return len(self.imgs) - self.seq_size + 1
        return len(self.imgs)//self.seq_size

    def __getitem__(self, index):
        imgs = torch.Tensor(self.seq_size, 3, self.input_dim, self.input_dim) #1x3x128x128
        targets = torch.LongTensor(self.seq_size, self.input_dim, self.input_dim) #1x128x128
        pil_imgs = []
        pil_targets = []

        for i in range(self.seq_size):
            if self.sliding_window == False:
                path = self.imgs[(index)*self.seq_size+i]
            else:
                path = self.imgs[index + i]
            #l'img ha un nome diverso dalla maschera, cambio il nome
            #ISBI2015
            list_path = path.split('\\')
            name_img = list_path[-1].replace('mask1', 'flair_pp')
            path_img = path.replace(list_path[-1], name_img)
            path_img = path_img.replace(self.split + 'annot', self.split)

            img = self.loader(path_img)
            target = self.loader(path)
            pil_imgs.append(img)
            pil_targets.append(target)

        for i in range(self.seq_size):
            if self.joint_transform is not None: #trasformazioni immagini + maschere
                transformed = self.joint_transform(pil_imgs + pil_targets)
            else:
                transformed = pil_imgs + pil_targets

            img = transformed[i]
                 
            if self.transform is not None: #trasformazioni immagini
                img = self.transform(img)
            
            m1 = img.min()
            m2 = img.max()
            if m1 != m2:
                img = (img-m1)/(m2-m1)
                img = (img-self.mean)/self.std

            imgs[i] = img
            target = transformed[self.seq_size+i]

            if self.target_transform is not None:
                target = self.target_transform(target)
                
            targets[i] = target
        
        return imgs, targets#.squeeze(1)
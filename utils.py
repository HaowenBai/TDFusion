import numpy as np
import cv2
import os
from skimage.io import imsave
import torch.nn.functional as F
import torch.nn as nn
import torch

def image_read(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode in ['RGB','GRAY','YCrCb'], 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img

def img_save(image, imagename, savepath, CrCb=None):
    temp = np.squeeze(image)
    path1 = os.path.join(savepath, 'RGB')
    path2 = os.path.join(savepath, 'Gray')
    if not os.path.exists(path2):
        os.makedirs(path2)
    imsave(os.path.join(path2, "{}.png".format(imagename)), temp) # If error, make sure imageio==2.26.0 is installed.

    if CrCb is not None:
        assert len(CrCb.shape) == 3 and CrCb.shape[2] == 2, "CrCb error"
        temp_RGB = cv2.cvtColor(np.concatenate((temp[..., np.newaxis], CrCb), axis=2), cv2.COLOR_YCrCb2RGB)
        if not os.path.exists(path1):
            os.makedirs(path1)
        temp_RGB[temp_RGB<0]=0
        temp_RGB[temp_RGB>255]=255
        imsave(os.path.join(path1, "{}.png".format(imagename)), temp_RGB)

def fuse_CrCb(CrCb1,CrCb2):
    assert len(CrCb1.shape) == 3 and CrCb1.shape[2] == 2, "CrCb error"
    assert len(CrCb2.shape) == 3 and CrCb2.shape[2] == 2, "CrCb error"
    Cf=(CrCb1*np.abs(CrCb1-0.5)+CrCb2*np.abs(CrCb2-0.5))/(np.abs(CrCb1-0.5)+np.abs(CrCb2-0.5)+1e-4)
    return Cf

def is_grayscale(image):
    return np.all(image[:,:,0] == image[:,:,1]) and np.all(image[:,:,1] == image[:,:,2])

def CE_Loss(inputs, target):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss  = nn.CrossEntropyLoss()(temp_inputs, temp_target)
    return CE_loss

def Fusionloss_int(img_F,img_A,img_B,w_A,w_B): 
    return F.mse_loss(w_A*(img_A-img_F),torch.zeros_like(img_F))+F.mse_loss(w_B*(img_B-img_F),torch.zeros_like(img_F))
   

def Fusionloss_grad(img_F,img_A,img_B):

    def sobel_filter(tensor):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        sobel_x = sobel_x.repeat(1, tensor.shape[1], 1, 1).to(tensor.device)
        sobel_y = sobel_y.repeat(1, tensor.shape[1], 1, 1).to(tensor.device)

        grad_x = F.conv2d(tensor, sobel_x, padding=1)
        grad_y = F.conv2d(tensor, sobel_y, padding=1)
        
        return grad_x, grad_y
    
    sobelx_f,sobely_f=sobel_filter(img_F)
    sobelx_a,sobely_a=sobel_filter(img_A)
    sobelx_b,sobely_b=sobel_filter(img_B)

    sobelx_max=(torch.abs(sobelx_a) >= torch.abs(sobelx_b))*sobelx_a+(torch.abs(sobelx_a) < torch.abs(sobelx_b))*sobelx_b
    sobely_max=(torch.abs(sobely_a) >= torch.abs(sobely_b))*sobely_a+(torch.abs(sobely_a) < torch.abs(sobely_b))*sobely_b

    return F.l1_loss(sobelx_f,sobelx_max)+F.l1_loss(sobely_f,sobely_max)
        
def tensor_resize(x,h=256,w=256):
    return F.interpolate(x, size=(h, w), mode='bilinear')

def named_params(curr_module, prefix=''): 
    memo=set()
    if hasattr(curr_module, 'named_leaves'):
        for name, p in curr_module.named_leaves():
            if p is not None and p not in memo:
                memo.add(p)
                yield prefix + ('.' if prefix else '') + name, p

    for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in named_params(module, submodule_prefix):
                yield name, p

def set_param(curr_mod, name, param):
    if '.' in name:
        n = name.split('.')
        module_name = n[0]
        rest = '.'.join(n[1:])
        for name, mod in curr_mod.named_children():
            if module_name == name:
                set_param(mod, rest, param)
                break
    else:
        setattr(curr_mod, name, param)

def inner_update(curr_mod,lr):
    for name, p  in  named_params(curr_mod):
        set_param(curr_mod, name+"_meta", p-lr*p.grad)

def outer_update(curr_mod,lr):
    with torch.no_grad():
        max_grad=0
        for _, param in curr_mod.named_parameters():
            max_grad=max(torch.max(torch.abs(param.grad)).item(),max_grad)
        if max_grad >0:  
            for _, param in curr_mod.named_parameters():
                param.add_(-lr*(param.grad/max_grad))
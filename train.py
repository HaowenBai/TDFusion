# -*- coding: utf-8 -*-

'''
------------------------------------------------------------------------------
Import packages
------------------------------------------------------------------------------
'''
import os
import torch.nn.functional as F
import warnings
import time
import torch
from torch.utils.data import DataLoader
from numpy import mean
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings('ignore')

from utils import CE_Loss,inner_update, outer_update, Fusionloss_int, Fusionloss_grad
from nets.Segformer import SegFormer
from nets.ReFusion import ReFusion,LPN
from nets.yolo_training import yoloLoss
from nets.yolo import YoloBody
from data.dataloader import Trainset_Seg, Trainset_Det, yolo_dataset_collate

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dataset selection
dataset="FMB"

if dataset in ["FMB"]:
    class_num=14
    resize_h=300
    resize_w=400
elif dataset in ['MSRS']:
    class_num=8
    resize_h=240
    resize_w=320
elif dataset in ["M3FD"]:
    class_num=6
    resize_h=192
    resize_w=256
elif dataset in ["LLVIP"]:
    class_num=1
    resize_h=256
    resize_w=320

# Parameter Configuration

num_epochs = 40
max_meta_step=200 

lr = 1e-4
step_size=10               
gamma=0.1
batch_size = 2

coef_f_int=1
coef_f_grad=1

Switch=1
if Switch==1:
    batch_size=batch_size//2

# Model Definition

Fusion_model= ReFusion().to(device)

if dataset in ["FMB","MSRS"]:
    Task_model = SegFormer(class_num + 1).to(device)
elif dataset in ["M3FD","LLVIP"]:  
    Task_model = YoloBody( class_num, phi="n", pretrained=False).to(device)
    yolo_loss = yoloLoss(Task_model)

Loss_model= torch.nn.DataParallel(LPN()).to(device)  

optimizer1 = torch.optim.Adam(Fusion_model.parameters(), lr=lr, weight_decay=0)
scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=step_size, gamma=gamma)

optimizer2 = torch.optim.Adam(Task_model.parameters(), lr=lr, weight_decay=0)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=step_size, gamma=gamma)

optimizer3 = torch.optim.Adam(Loss_model.parameters(), lr=lr, weight_decay=0,eps=1e-10)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=step_size, gamma=gamma)

if dataset in ["FMB","MSRS"]:
    taskloader = DataLoader(Trainset_Seg(dataset),batch_size=batch_size, shuffle=True, num_workers=0)
elif dataset in ["M3FD","LLVIP"]:
    taskloader = DataLoader(Trainset_Det(dataset), shuffle = True, batch_size = batch_size,collate_fn=yolo_dataset_collate)

exppath=os.path.join("exp",time.strftime("%m_%d_%H_%M", time.localtime()))
os.makedirs(exppath,exist_ok=True)
os.makedirs(os.path.join(exppath,'model'),exist_ok=True)

# Model Definition
torch.backends.cudnn.benchmark = True

Fusion_model.train()
Task_model.train()
Loss_model.train()
pbar = tqdm(total=len(taskloader))

for epoch in range(num_epochs):
    lr_epoch=optimizer1.param_groups[0]['lr']
    F_loss=[]
    F_loss_int=[]
    F_loss_grad=[]
    T_loss=[]

    # Learning of G

    pbar.total=max_meta_step
    Loss_model.train()

    for i, (data_IR_ori, data_VIS_ori, mask_t, index) in enumerate(taskloader):
        data_IR_ori, data_VIS_ori,mask_t= data_IR_ori.cuda(), data_VIS_ori.cuda(),mask_t.cuda() 
        H,W=data_IR_ori.shape[2:]

        if i==max_meta_step*2:
            break

        # inner loop
        if i %2 ==0:
            h=int((H-256)*random.random())
            w=int((W-256)*random.random())
            torch.save(Task_model.state_dict(), os.path.join(exppath,'interim_sta_DS.pth'))

            data_IR_crop=data_IR_ori[:,:,h:h+256,w:w+256]
            data_VIS_crop=data_VIS_ori[:,:,h:h+256,w:w+256]

            data_IR_resize=F.interpolate(data_IR_ori, size=(resize_h, resize_w), mode='bilinear')
            data_VIS_resize=F.interpolate(data_VIS_ori, size=(resize_h, resize_w), mode='bilinear')                  

            if Switch:
                data_IR_crop,data_VIS_crop=torch.cat((data_IR_crop,data_VIS_crop),0),torch.cat((data_VIS_crop,data_IR_crop),0)
                data_IR_resize,data_VIS_resize=torch.cat((data_IR_resize,data_VIS_resize),0),torch.cat((data_VIS_resize,data_IR_resize),0)
                index=torch.cat((index,index),0)
                mask_t=torch.cat((mask_t,mask_t),0)
                if dataset in ["M3FD","LLVIP"]:
                    mask_t[-mask_t.shape[0]//2:,0]=mask_t[-mask_t.shape[0]//2:,0]+batch_size # Image Index
                    
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer3.zero_grad()

            data_Fuse=Fusion_model(torch.cat((data_IR_crop,data_VIS_crop),1))
            w=Loss_model(torch.cat((data_IR_crop,data_VIS_crop),1))

            loss_f_int=Fusionloss_int(data_Fuse,data_IR_crop,data_VIS_crop,w[:,0:1,:,:],w[:,1:2,:,:])
            loss_f_grad=Fusionloss_grad(data_Fuse,data_IR_crop,data_VIS_crop)

            Fusion_loss=coef_f_int*loss_f_int + coef_f_grad*loss_f_grad
            Fusion_loss.backward(create_graph=True,retain_graph=True)

            inner_update(Fusion_model,lr_epoch)  #update F as F'  
            
            w=Loss_model(torch.cat((data_IR_resize,data_VIS_resize),1)).detach()
            data_Fuse=Fusion_model(torch.cat((data_IR_resize,data_VIS_resize),1)).detach()
            data_task=Task_model(data_Fuse)

            if dataset in ["FMB","MSRS"]:
                Task_loss=CE_Loss(data_task, mask_t)
            elif dataset in ["M3FD","LLVIP"]:
                Task_loss=yolo_loss(data_task, mask_t)

            Task_loss.backward()
            optimizer2.step()

        # outer loop 
        else: 
            data_IR_resize=F.interpolate(data_IR_ori, size=(resize_h, resize_w), mode='bilinear')
            data_VIS_resize=F.interpolate(data_VIS_ori, size=(resize_h, resize_w), mode='bilinear')   

            if Switch:
                data_IR_resize,data_VIS_resize=torch.cat((data_IR_resize,data_VIS_resize),0),torch.cat((data_VIS_resize,data_IR_resize),0)
                index=torch.cat((index,index),0)
                mask_t=torch.cat((mask_t,mask_t),0)
                if dataset in ["M3FD","LLVIP"]:
                    mask_t[-mask_t.shape[0]//2:,0]=mask_t[-mask_t.shape[0]//2:,0]+batch_size
                    
            optimizer3.zero_grad()

            data_Fuse=Fusion_model(torch.cat((data_IR_resize,data_VIS_resize),1),meta=True)
            data_task=Task_model(data_Fuse)

            if dataset in ["FMB","MSRS"]:
                Task_loss=CE_Loss(data_task, mask_t)
            elif dataset in ["M3FD","LLVIP"]:
                Task_loss=yolo_loss(data_task, mask_t)
            Task_loss.backward()

            outer_update(Loss_model,lr_epoch)

            Task_model.load_state_dict(torch.load(os.path.join(exppath,'interim_sta_DS.pth'))) 

            F_loss.append(Fusion_loss.item())
            T_loss.append(Task_loss.item())
            F_loss_int.append(loss_f_int.item())
            F_loss_grad.append(loss_f_grad.item())
            pbar.update(1)

        pbar.set_description("Learning of G  Epoch: %d/%d, FusionLoss: %f, F_int: %f, F_grad: %f, TaskLoss: %f"% (epoch + 1, num_epochs , mean(F_loss),mean(F_loss_int),mean(F_loss_grad),mean(T_loss)))
        
    logfile = open(os.path.join(exppath,"log.txt"),'a')
    logfile.write("Learning of G  Epoch: %d/%d, FusionLoss: %f, F_int: %f, F_grad: %f, TaskLoss: %f"% (epoch + 1, num_epochs , mean(F_loss),mean(F_loss_int),mean(F_loss_grad),mean(T_loss))+'\n')
    logfile.close()

    pbar.reset()  

    # Learning of F and T
    Loss_model.eval()

    F_loss=[]
    F_loss_int=[]
    F_loss_grad=[]
    T_loss=[]

    pbar.total=len(taskloader)

    for i, (data_IR_ori, data_VIS_ori, mask_t,index) in enumerate(taskloader):
        data_IR_ori, data_VIS_ori,mask_t= data_IR_ori.cuda(), data_VIS_ori.cuda(),mask_t.cuda()
        H,W=data_IR_ori.shape[2:]
        h=int((H-256)*random.random())
        w=int((W-256)*random.random())

        data_IR_crop=data_IR_ori[:,:,h:h+256,w:w+256]
        data_VIS_crop=data_VIS_ori[:,:,h:h+256,w:w+256]

        data_IR_resize=F.interpolate(data_IR_ori, size=(resize_h, resize_w), mode='bilinear')
        data_VIS_resize=F.interpolate(data_VIS_ori, size=(resize_h, resize_w), mode='bilinear')   
            
        if Switch:
            data_IR_crop,data_VIS_crop=torch.cat((data_IR_crop,data_VIS_crop),0),torch.cat((data_VIS_crop,data_IR_crop),0)
            data_IR_resize,data_VIS_resize=torch.cat((data_IR_resize,data_VIS_resize),0),torch.cat((data_VIS_resize,data_IR_resize),0)
            index=torch.cat((index,index),0)
            mask_t=torch.cat((mask_t,mask_t),0)
            if dataset in ["M3FD","LLVIP"]:
                mask_t[-mask_t.shape[0]//2:,0]=mask_t[-mask_t.shape[0]//2:,0]+batch_size

        # update F
        optimizer1.zero_grad()
        data_Fuse=Fusion_model(torch.cat((data_IR_crop,data_VIS_crop),1))
        
        w=Loss_model(torch.cat((data_IR_crop,data_VIS_crop),1)).detach()

        loss_f_int=Fusionloss_int(data_Fuse,data_IR_crop,data_VIS_crop,w[:,0:1,:,:],w[:,1:2,:,:])
        loss_f_grad=Fusionloss_grad(data_Fuse,data_IR_crop,data_VIS_crop)

        Fusion_loss=coef_f_int*loss_f_int + coef_f_grad*loss_f_grad 
        Fusion_loss.backward()
        optimizer1.step() 

        # update T
        optimizer2.zero_grad()
        w=Loss_model(torch.cat((data_IR_resize,data_VIS_resize),1)).detach()

        data_Fuse=Fusion_model(torch.cat((data_IR_resize,data_VIS_resize),1)).detach()
        data_task=Task_model(data_Fuse)
        if dataset in ["FMB","MSRS"]:
            Task_loss=CE_Loss(data_task, mask_t)
        elif dataset in ["M3FD","LLVIP"]:
            Task_loss=yolo_loss(data_task, mask_t)
        Task_loss.backward()
        optimizer2.step() 

        F_loss.append(Fusion_loss.item())
        F_loss_int.append(loss_f_int.item())
        F_loss_grad.append(loss_f_grad.item())
        T_loss.append(Task_loss.item())

        pbar.update(1)
        pbar.set_description("Learning of F  Epoch: %d/%d, FusionLoss: %f, F_int: %f, F_grad: %f, TaskLoss: %f"% (epoch + 1, num_epochs , mean(F_loss),mean(F_loss_int),mean(F_loss_grad),mean(T_loss)))
        
    logfile = open(os.path.join(exppath,"log.txt"),'a')
    logfile.write("Learning of F  Epoch: %d/%d, FusionLoss: %f, F_int: %f, F_grad: %f, TaskLoss: %f"% (epoch + 1, num_epochs , mean(F_loss),mean(F_loss_int),mean(F_loss_grad),mean(T_loss))+'\n')
    logfile.close()

    pbar.reset()

    # adjust the learning rate and save model
    scheduler1.step()  
    scheduler2.step()
    scheduler3.step()

    if epoch % 1 == 0:
        torch.save(Fusion_model.state_dict(), os.path.join(exppath,'model', 'ckpt_%s.pth' % (str(epoch+1))))


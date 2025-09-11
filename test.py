from tqdm import tqdm
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.CRITICAL)
from utils import *
from nets.ReFusion import ReFusion
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''Select the desired model'''
path_model=r"models/TDFusion_MSRS.pth"
# path_model=r"models/TDFusion_FMB.pth"
# path_model=r"models/TDFusion_M3FD.pth"
# path_model=r"models/TDFusion_LLVIP.pth"

path_img1=r"test_cases/infrared"
path_img2=r"test_cases/visible"
path_result=r"test_results"

Fusion_model= ReFusion().to(device)
Fusion_model.load_state_dict(torch.load(path_model))
os.makedirs(path_result,exist_ok=True)
with torch.no_grad():
    for imgname in tqdm(os.listdir(path_img1)):
        img1=image_read(os.path.join(path_img1, imgname))
        img2=image_read(os.path.join(path_img2, imgname))

        img1_YCrCb=cv2.cvtColor(img1, cv2.COLOR_RGB2YCrCb)
        img2_YCrCb=cv2.cvtColor(img2, cv2.COLOR_RGB2YCrCb)

        if is_grayscale(img1):
            if is_grayscale(img2):
                CrCb=None
            else:
                CrCb=img2_YCrCb[:,:,1:]
        else:
            if is_grayscale(img2): 
                CrCb=img1_YCrCb[:,:,1:]
            else:
                CrCb=fuse_CrCb(img1_YCrCb[:,:,1:],img2_YCrCb[:,:,1:])

        img1=img1_YCrCb[:,:,0][np.newaxis,np.newaxis,...]/255
        img2=img2_YCrCb[:,:,0][np.newaxis,np.newaxis,...]/255
        img1 = ((torch.FloatTensor(img1))).to(device)
        img2 = ((torch.FloatTensor(img2))).to(device)

        data_Fuse=Fusion_model(torch.cat((img1,img2),1))
        data_Fuse=(data_Fuse-torch.min(data_Fuse))/(torch.max(data_Fuse)-torch.min(data_Fuse))
        fused_image = np.squeeze((data_Fuse * 255).cpu().numpy())

        img_save(fused_image, imgname.split(sep='.')[0], path_result, CrCb)

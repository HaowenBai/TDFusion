# TDFusion
Codes for ***Task-driven Image Fusion with Learnable Fusion Loss (CVPR 2025 highlight)***

[Haowen Bai](https://haowenbai.github.io/), [Jiangshe Zhang](http://gr.xjtu.edu.cn/web/jszhang), [Zixiang Zhao](https://zhaozixiang1228.github.io/), [Yichen Wu](https://wuyichen-97.github.io/), [Lilun Deng](), [Yukun Cui](), [Tao Feng](), [Shuang Xu](https://shuangxu96.github.io/).

-[*[Paper]*](https://openaccess.thecvf.com/content/CVPR2025/html/Bai_Task-driven_Image_Fusion_with_Learnable_Fusion_Loss_CVPR_2025_paper.html0)  
-[*[ArXiv]*](https://arxiv.org/pdf/2412.03240)  


## Abstract

Multi-modal image fusion aggregates information from multiple sensor sources, achieving superior visual quality and perceptual features compared to single-source images, often improving downstream tasks. However, current fusion methods for downstream tasks still use predefined fusion objectives that potentially mismatch the downstream tasks, limiting adaptive guidance and reducing model flexibility. To address this, we propose Task-driven Image Fusion (TDFusion), a fusion framework incorporating a learnable fusion loss guided by task loss. Specifically, our fusion loss includes learnable parameters modeled by a neural network called the loss generation module. This module is supervised by the downstream task loss in a meta-learning manner. The learning objective is to minimize the task loss of fused images after optimizing the fusion module with the fusion loss. Iterative updates between the fusion module and the loss module ensure that the fusion network evolves toward minimizing task loss, guiding the fusion process toward the task objectives. TDFusion's training relies entirely on the downstream task loss, making it adaptable to any specific task. It can be applied to any architecture of fusion and task networks. Experiments demonstrate TDFusion's performance through fusion experiments conducted on four different datasets, in addition to evaluations on semantic segmentation and object detection tasks.


## Update
- [2025/3] Release the code.

## Citation

```
@inproceedings{bai2024task,
  title={Task-driven Image Fusion with Learnable Fusion Loss},
  author={Bai, Haowen and Zhang, Jiangshe and Zhao, Zixiang and Wu, Yichen and Deng, Lilun and Cui, Yukun and Feng, Tao and Xu, Shuang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {7457-7468}
  year={2025}
}
```


## 🌐 Usage

### 🏊 Training
**1. Data Preparation**

Download the data and place the data according to the path in ``'./data/dataloader.py'``.

[FMB download link](https://github.com/JinyuanLiu-CV/SegMiF) (Place the data according to lines 14-16.)

[MSRS download link](https://github.com/Linfeng-Tang/MSRS)  (Place the data according to lines 18-20.)

[M3FD download link](https://github.com/JinyuanLiu-CV/TarDAL) (Place the data according to lines 44-45.)

[LLVIP download link](https://github.com/bupt-ai-cz/LLVIP) (Place the data according to lines 50-51.)

Our partitions for M3FD and LLVIP are stored in ``./data/M3FD_train.txt``, ``./data/M3FD_test.txt``, ``./data/LLVIP_train.txt``, and ``./data/LLVIP_test.txt``.

**2. Commence Training**

Modify the variable 'dataset' on line 31 of ``train.py``, and run
```
python train.py
``` 
The logs and models will be saved in the ``./exp`` directory based on the timestamp.


### 🏄 Testing

**1. Pretrained models**

Pretrained models are available in ``./models/TDFusion_MSRS.pth`` , ``./models/TDFusion_FMB.pth``, ``./models/TDFusion_M3FD.pth`` and ``./models/TDFusion_LLVIP.pth``, which are the models trained on MSRS, FMB, M3FD and LLVIP, respectively.

**2. Test cases**

The 'test_cases' folder contains four examples that appear in the main paper.
Running 
```
python test.py
``` 
will fuse these cases, and the fusion results will be saved in the 'test_results' folder.

**3. Test customization**

Modify the variables in ``test.py`` as needed: 

'path_model' (the path of the pretrained model), 

'path_img1' (the path of the infrared images), 

'path_img2' (the path of the visible images), 

'path_result' (the path to save the fusion result). 

Then run
```
python test.py
``` 
and the fusion results will be saved in the 'path_result'.

### About Downstream tasks

Our training and testing code is from the following GitHub repository:
https://github.com/bubbliiiing

Specifically:  
Semantic Segmentation: https://github.com/bubbliiiing/segformer-pytorch   
Object Detection: https://github.com/bubbliiiing/yolov8-pytorch  
We have not made any modifications to the code, except for hyperparameters such as epoch and batch size. Please refer to the original repositories for details.

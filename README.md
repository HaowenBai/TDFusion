# TDFusion
Codes for ***Task-driven Image Fusion with Learnable Fusion Loss (CVPR 2025)***

[Haowen Bai](), [Jiangshe Zhang](http://gr.xjtu.edu.cn/web/jszhang), [Zixiang Zhao](https://zhaozixiang1228.github.io/), [Yichen Wu](), [Lilun Deng](), [Yukun Cui](), [Tao Feng](), [Shuang Xu](https://shuangxu96.github.io/).

-[*[ArXiv]*](https://arxiv.org/pdf/2412.03240)  


## Abstract

Multi-modal image fusion aggregates information from multiple sensor sources, achieving superior visual quality and perceptual characteristics compared to any single source, often enhancing downstream tasks. However, current fusion methods for downstream tasks still use predefined fusion objectives that potentially mismatch the downstream tasks, limiting adaptive guidance and reducing model flexibility. To address this, we propose Task-driven Image Fusion (TDFusion), a fusion framework incorporating a learnable fusion loss guided by task loss. Specifically, our fusion loss includes learnable parameters modeled by a neural network called the loss generation module. This module is supervised by the loss of downstream tasks in a meta-learning manner. The learning objective is to minimize the task loss of the fused images, once the fusion module has been optimized by the fusion loss. Iterative updates between the fusion module and the loss module ensure that the fusion network evolves toward minimizing task loss, guiding the fusion process toward the task objectives. TDFusion’s training relies solely on the loss of downstream tasks, making it adaptable to any specific task. It can be applied to any architecture of fusion and task networks. Experiments demonstrate TDFusion’s performance in both fusion and task-related applications, including four public fusion datasets, semantic segmentation, and object detection.


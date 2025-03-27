import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import os
import torch.utils.data as Data
from utils import image_read

class Trainset_Seg(Data.Dataset):

    def __init__(self,dataname):
        
        if dataname is "FMB":
            self.IR_path=r"data\FMB\train\ir"
            self.VI_path=r"data\FMB\train\vi"
            self.mask_path=r"data\FMB\train\label"
        elif dataname is "MSRS":
            self.IR_path=r"data\MSRS\train\ir"
            self.VI_path=r"data\MSRS\train\vi"
            self.mask_path=r"data\MSRS\train\label"
        else:
            print("unknown data")

        self.file_name_list = os.listdir(self.IR_path)

    def __len__(self):
        return len(self.file_name_list)
    
    def __getitem__(self, index):
        IR=image_read(os.path.join(self.IR_path, self.file_name_list[index]), 'GRAY')[np.newaxis,...]/255.
        VI=image_read(os.path.join(self.VI_path, self.file_name_list[index]), 'GRAY')[np.newaxis,...]/255.
        segmask=image_read(os.path.join(self.mask_path, self.file_name_list[index]), 'GRAY').astype(int)
        return  torch.Tensor(IR),torch.Tensor(VI),torch.Tensor(segmask).long(),index
    
def cvtColor(image):
    image = image.convert('L')
    return image 

class Trainset_Det(Dataset):
    def __init__(self, data_name):
        super(Trainset_Det, self).__init__()
        if data_name=="M3FD":
            self.annotation_lines=self.get_annotation(r"data\M3FD_train.txt")
            self.ir_path=r"data/M3FD/ir"
            self.vi_path=r"data/M3FD/vi"
            self.h=768
            self.w=1024
        elif data_name=="LLVIP":
            self.annotation_lines=self.get_annotation(r"data\LLVIP_train.txt")
            self.ir_path=r"data/LLVIP/ir"
            self.vi_path=r"data/LLVIP/vi"
            self.h=1024
            self.w=1280
        else:
            print("unknown data")

        self.length             = len(self.annotation_lines)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        line=self.annotation_lines[index].split()

        ir=cvtColor(Image.open(os.path.join(self.ir_path,line[0])))
        vi=cvtColor(Image.open(os.path.join(self.vi_path,line[0])))

        iw, ih  = ir.size 

        box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        if iw!=self.w or ih!=self.h:
            ir       = ir.resize((self.w,self.h), Image.BICUBIC)
            vi       = vi.resize((self.w,self.h), Image.BICUBIC)


        ir       = np.array(ir, dtype=np.float32)[None,...]/255.0
        vi       = np.array(vi, dtype=np.float32)[None,...]/255.0
        box         = np.array(box, dtype=np.float32)
        np.random.shuffle(box)

        nL          = len(box)
        labels_out  = np.zeros((nL, 6))
        if nL:

            box[:, [0, 2]] = box[:, [0, 2]] / self.h
            box[:, [1, 3]] = box[:, [1, 3]] / self.w

            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
            
            labels_out[:, 1] = box[:, -1]
            labels_out[:, 2:] = box[:, :4]
            
        return ir,vi, labels_out,index


    def get_annotation(self,annotation_path):
            with open(annotation_path, encoding='utf-8') as f:
                train_lines = f.readlines()
                return train_lines
            
    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        line    = annotation_line.split()

        image   = Image.open(line[0])
        image   = cvtColor(image)

        iw, ih  = image.size
        h, w    = input_shape

        box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box

            return image_data, box
                
def yolo_dataset_collate(batch):
    ir_images=[]
    vi_images  = []
    bboxes  = []
    index_list=[]
    for i, (ir,vi, box,index) in enumerate(batch):
        ir_images.append(ir)
        vi_images.append(vi)
        box[:, 0] = i
        bboxes.append(box)
        index_list.append(index)  
    ir_images  = torch.from_numpy(np.array(ir_images)).type(torch.FloatTensor)
    vi_images  = torch.from_numpy(np.array(vi_images)).type(torch.FloatTensor)
    bboxes  = torch.from_numpy(np.concatenate(bboxes, 0)).type(torch.FloatTensor)
    index_list=torch.tensor(index_list)
    return ir_images, vi_images ,bboxes,index_list


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.getcwd())
    dataset = Trainset_Det(r"data\M3FD_train.txt")
    a=dataset[0]
    print(a)
    print(123)
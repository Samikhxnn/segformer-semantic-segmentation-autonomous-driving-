import torch
from torch.utils.data import Dataset
import os
import cv2



class Cp(Dataset):
    def __init__(self,image_dir,mask_dir,transform=None):

        self.images=[]
        self.masks=[]

        self.transform=transform


        for img in os.listdir(image_dir):

            img_path=os.path.join(image_dir,img)
            mask_path=os.path.join(mask_dir,img)
            self.images.append(img_path)
            self.masks.append(mask_path)



    def __len__(self):
        return len(self.images)


    def __getitem__(self,idx):
        image=cv2.imread(self.images[idx])
        mask=cv2.imread(self.masks[idx],0)

        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed=self.transform(
                image=image,
                mask=mask
            )
            image=transformed['image']
            mask=transformed['mask']

        return image,mask    

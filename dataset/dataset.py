import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np

id_to_trainid = {
    7: 0,
    8: 1,
    11: 2,
    12: 3,
    13: 4,
    17: 5,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    31: 16,
    32: 17,
    33: 18,
}



class  CityScrap(Dataset):
    def __init__(self,image_dir,mask_dir,transform=None):

        self.images=[]
        self.masks=[]
        self.transform=transform


        for city in os.listdir(image_dir):
            img_path=os.path.join(image_dir,city)
            mask_path=os.path.join(mask_dir,city)

            for file in os.listdir(img_path):
                self.images.append(os.path.join(img_path,file))

                mask=file.replace(
                        "leftImg8bit",
                        "gtFine_labelIds")
                
                self.masks.append(os.path.join(mask_path,mask))

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):


        image=cv2.imread(self.images[idx])

        mask=cv2.imread(self.masks[idx],0)

        mask = np.array(mask)

        new_mask = np.full(mask.shape, 255, dtype=np.uint8)

        for k, v in id_to_trainid.items():
            new_mask[mask == k] = v

        mask = new_mask

        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        if self.transform:

            transformed=self.transform(
                image=image,
                mask=mask
            )
        
        image=transformed["image"]
        mask=transformed["mask"]


        return image,mask

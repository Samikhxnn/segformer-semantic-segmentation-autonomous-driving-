import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F


from dataset import CityScrap
from transform import get_val_transform
from model import Model
from metrics import calculate_iou

from config import *


id_to_trainid = {
    7:0,    # road
    8:1,    # sidewalk
    11:2,   # building
    12:3,   # wall
    13:4,   # fence
    17:5,   # pole
    19:6,   # traffic light
    20:7,   # traffic sign
    21:8,   # vegetation
    22:9,   # terrain
    23:10,  # sky
    24:11,  # person
    25:12,  # rider
    26:13,  # car
    27:14,  # truck
    28:15,  # bus
    31:16,  # train
    32:17,  # motorcycle
    33:18,  # bicycle
}



val_data= CityScrap(val_image_dir,val_mask_dir,transform=get_val_transform(256))

val_loader=DataLoader(val_data,
                      batch_size=4,
                      shuffle=False)



model=Model(num_classes=19)

model.load_state_dict(torch.load("best_model.pth"))


model.to(DEVICE)

model.eval()


with torch.no_grad():



        score=[]
        for image ,mask in val_loader:

                image=image.to(DEVICE)
                mask=mask.to(DEVICE)

                output=model(image)

                logits=output.logits

                logits = F.interpolate(
                        logits,
                        size=mask.shape[-2:],      # (256,256)
                        mode="bilinear",
                        align_corners=False
                        )


                iou=calculate_iou(logits,mask,num_classes=19)

                score.append(iou)

        print(f"Mean IOU: {torch.mean(torch.stack(score))}")
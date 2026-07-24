import torch
from torch.utils.data import DataLoader

from dataset import CityScrap
from transform import get_train_transform
from model import Model
from loss import DiceCELoss


from config import (
                    train_image_dir,
                    train_mask_dir,
                    DEVICE
)




train_dataset=CityScrap(train_image_dir,train_mask_dir,transform=get_train_transform(size=256))


train_loader=DataLoader(train_dataset,
                        batch_size=6,
                        shuffle=True,
                        )


model=Model(num_classes=19).to(DEVICE)

# set up loss and optimiser
criterion=DiceCELoss()
optimizer=torch.optim.AdamW(model.parameters(),lr=0.0001)

best_loss = float("inf")

for epoch in range(30):

    model.train()
    total_loss=0

    for image,mask in train_loader:

        image=image.to(DEVICE)
        mask=mask.to(DEVICE)


        

        output=model(image)

        logits=output.logits

        logits = torch.nn.functional.interpolate(
            logits,
        size=mask.shape[-2:],
        mode="bilinear",
        align_corners=False
            )

        mask = mask.long()

        
        
        loss=criterion(logits,mask)
        total_loss+=loss.item()


    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Epoch [{epoch+1}/30]")
    print(f"Loss: {total_loss/len(train_loader)}")

    avg_loss = total_loss / len(train_loader)

    if avg_loss < best_loss:

        best_loss = avg_loss

        torch.save(
            model.state_dict(),
            "best_model.pth"
        )

        print("Best model saved!")
    
        








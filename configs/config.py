import torch


#-----------------------------
# train roots
#-----------------------------

train_image_dir=r"C:\Users\Asus\Desktop\cityscapes\leftimg8bit\train"
train_mask_dir=r"C:\Users\Asus\Desktop\cityscapes\gtfine\train"


#-----------------------------
# validation roots
#-----------------------------

val_image_dir=r"C:\Users\Asus\Desktop\cityscapes\leftimg8bit\val"
val_mask_dir=r"C:\Users\Asus\Desktop\cityscapes\gtfine\val"


#-----------------------------
# device
#-----------------------------

DEVICE="cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")



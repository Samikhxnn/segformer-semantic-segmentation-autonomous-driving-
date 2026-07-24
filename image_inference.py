import torch
import cv2
import numpy as np
import torch.nn.functional as F

from model import Model
from config import DEVICE
from transform import get_val_transform


NUM_CLASSES = 19


# -----------------------
# Load Model
# -----------------------

model = Model(num_classes=NUM_CLASSES)

checkpoint = torch.load(
    "model.path",
    map_location=DEVICE
)

model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))

model.to(DEVICE)
model.eval()


# -----------------------
# Transform
# -----------------------

transform = get_val_transform(size=512)


# -----------------------
# Color Palette
# -----------------------

np.random.seed(42)

colors = np.random.randint(
    0,
    255,
    (NUM_CLASSES,3),
    dtype=np.uint8
)


# -----------------------
# Input Image
# -----------------------

image_path = r"C:\Users\Asus\Downloads\img4.jpg"

image = cv2.imread(image_path)

original = image.copy()

h,w,_ = image.shape


rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


# -----------------------
# Preprocess
# -----------------------

data = transform(image=rgb)

input_image = data["image"]

input_image = input_image.unsqueeze(0).to(DEVICE)



# -----------------------
# Prediction
# -----------------------

with torch.no_grad():

    output = model(input_image)


    # If using HuggingFace SegFormer
    if hasattr(output, "logits"):
        logits = output.logits

    else:
        logits = output


    # Resize logits to input size
    logits = F.interpolate(
        logits,
        size=(512,512),
        mode="bilinear",
        align_corners=False
    )


    prediction = torch.argmax(
        logits,
        dim=1
    )


prediction = prediction.squeeze().cpu().numpy()



# -----------------------
# Resize mask back
# -----------------------

prediction = cv2.resize(
    prediction.astype(np.uint8),
    (w,h),
    interpolation=cv2.INTER_NEAREST
)



# -----------------------
# Create overlay
# -----------------------

mask_color = colors[prediction]


overlay = cv2.addWeighted(
    original,
    0.5,
    mask_color,
    0.5,
    0
)


# -----------------------
# Save
# -----------------------

cv2.imwrite(
    "segmentation_result.jpg",
    overlay
)


cv2.imshow(
    "Segmentation",
    overlay
)

cv2.waitKey(0)
cv2.destroyAllWindows()
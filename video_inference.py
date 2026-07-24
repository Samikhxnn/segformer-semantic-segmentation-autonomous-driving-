import cv2
import torch
import numpy as np

from model import Model
from transform import get_val_transform
from config import DEVICE

NUM_CLASSES = 19

# ------------------------
# Load model
# ------------------------

model = Model(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

transform = get_val_transform(size=512)

# ------------------------
# Random color palette
# ------------------------

np.random.seed(42)
colors = np.random.randint(0,255,(NUM_CLASSES,3),dtype=np.uint8)

# ------------------------
# Video
# ------------------------

video_path = r"C:\Users\Asus\Downloads\19908995-hd_1920_1080_25fps.mp4"

cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

writer = cv2.VideoWriter(
    "output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width,height)
)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    transformed = transform(image=rgb)
    image = transformed["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        output = model(image)
        logits = output.logits

        pred = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

    pred = cv2.resize(
        pred.astype(np.uint8),
        (width,height),
        interpolation=cv2.INTER_NEAREST
    )

    mask = colors[pred]

    overlay = cv2.addWeighted(frame,0.5,mask,0.5,0)

    writer.write(overlay)

    cv2.imshow("Prediction",overlay)

    if cv2.waitKey(1)==27:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
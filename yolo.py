import cv2
import torch
from torchvision import models, transforms
import json
import numpy as np

# Load Faster R-CNN model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
score_threshold = 0.7; 
# Load COCO class labels
with open('coco_labels.json', 'r') as f:
    class_labels = json.load(f)

# Preprocess function
def preprocess(image):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(image)
    return input_tensor

# Load the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Preprocess input
    input_tensor = preprocess(frame)
    input_batch = input_tensor.unsqueeze(0)

    # Predict bounding boxes and labels
    with torch.no_grad():
        predictions = model(input_batch)
    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']

    # Display results
    for box, label, score in zip(boxes, labels, scores):
        box = [int(coord) for coord in box]
        
        class_name = class_labels[str(label.item())]
        if score < score_threshold or label.item() != 1:
          continue

          
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        cv2.putText(frame, f"{class_name} {score:.2f}", (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow('Object Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()

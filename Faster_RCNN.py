from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

import cv2
import numpy as np
import matplotlib.pyplot as plt

def readimage(path):
  image = read_image(path)
  batch = [image]
  image = image.permute(1, 2, 0).numpy()
  return image,batch

def prediction(batch):
  # Step 1: Initialize model with the best available weights
  weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
  model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
  model.eval()

  # Step 2: Initialize the inference transforms
  preprocess = weights.transforms()
  # Step 4: Use the model and visualize the prediction
  prediction = model(batch)[0]
  boxes = prediction['boxes']
  labels = prediction["labels"]
  boxes = prediction["boxes"]
  categories = weights.meta["categories"]
  return boxes,labels,categories

def draw_boxes(image, boxes, labels,categories):
  for i in range(len(labels)):
    x1 = int(boxes[i][0].item()) # Convert x1 to integer
    y1 = int(boxes[i][1].item()) # Convert y1 to integer
    x2 = int(boxes[i][2].item()) # Convert x2 to integer
    y2 = int(boxes[i][3].item())
    cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.putText(image,categories[labels[i]],(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
  return image

def virsualize(image):
  plt.figure(figsize=(10,10))
  plt.imshow(image)
  plt.show()

import argparse
if __name__ == '__main__':
    parsel = argparse.ArgumentParser()
    parsel.add_argument('--image_path',type=str,default='image_2.jpg')
    args = parsel.parse_args()
    image,batch = readimage(args.image_path)
    boxes,labels,categories = prediction(batch)
    image_bbox = draw_boxes(image,boxes,labels,categories)






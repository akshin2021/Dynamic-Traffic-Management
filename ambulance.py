import cv2
import time
from ultralytics import YOLO
model=YOLO('best (2).pt')

image= model('ambulance.jpg',show=True)


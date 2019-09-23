# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 13:04:40 2019

@author: Gayatri
"""

from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()
os.chdir('F:/Internship_CV/ImageAI_Model')

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
#detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))

detector.setModelPath("resnet50_coco_best_v2.0.1.h5")
detector.loadModel()
custom_objects = detector.CustomObjects(person=True, car=False)
#detections = detector.detectCustomObjectsFromImage(input_image=os.path.join(execution_path , "image.png"), output_image_path=os.path.join(execution_path , "image_new.png"), custom_objects=custom_objects, minimum_percentage_probability=65)
detections = detector.detectCustomObjectsFromImage(input_image="Pedestrian6.jpg", output_image_path="Pedestrian6_new.jpg", custom_objects=custom_objects, minimum_percentage_probability=65)

for eachObject in detections:
   print(eachObject["name"] + " : " + eachObject["percentage_probability"] )
   print("--------------------------------")
   
   
   
from IPython.display import Image
Image("Pedestrian6_new.jpg")
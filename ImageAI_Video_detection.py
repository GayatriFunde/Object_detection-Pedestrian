# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 07:32:32 2019

@author: Gayatri
"""
#import cv2
import os
from imageai.Detection import VideoObjectDetection

os.chdir('F:\Internship_CV\ImageAI_Model')
#camera = cv2.VideoCapture(0)
#ret,frame = camera.read()
detector = VideoObjectDetection()

detector.setModelTypeAsRetinaNet() #performing your object detection tasks using the pre-trained “RetinaNet” model

#setModelPath() function accepts a string which must be the path to the model file you downloaded
# and must corresponds to the model type you set for your object detection instance.
detector.setModelPath("resnet50_coco_best_v2.0.1.h5")
detector.loadModel() # parameter detection_speed (optional)

video_path = detector.detectObjectsFromVideo(input_file_path="Video_Pedestrian.mp4",
    output_file_path ="Video_Pedestrian_detected_video_1", frames_per_second=20, log_progress=True, minimum_percentage_probability=30)

print(video_path)

#camera.release()
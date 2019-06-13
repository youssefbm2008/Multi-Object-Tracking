# -*- coding: utf-8 -*-
"""
Created on Fri May 24 16:44:43 2019

@author: Youssef
"""

import dlib # dlib for accurate face detection
import cv2 # opencv
import imutils # helper functions from pyimagesearch.com
from imutils.video import VideoStream
import argparse
import time

# Grab video from your webcam
stream = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
(grabbed, frame) = stream.read()
# resize the frames to be smaller and switch to gray scale
frame = imutils.resize(frame, width=700)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Make copies of the frame for transparency processing
overlay = frame.copy()
output = frame.copy()

# set transparency value
alpha  = 0.5

# detect faces in the gray scale frame
face_rects = detector(gray, 0)
        
print(face_rects)      
        
        
        

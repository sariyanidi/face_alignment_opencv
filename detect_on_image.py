#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 18:45:22 2021
@author: sariyanidi
"""

import cv2
import face_alignment_opencv as fao
import matplotlib.pyplot as plt
import os

d = fao.FaceDetector()
a = fao.FaceAligner()

im = cv2.imread('test.png')

(x0, y0, xf, yf) = d.get_single_detection(im)
p = a.get_landmarks(im, x0, y0, xf, yf)

plt.clf()
plt.gca().set_axis_off()
plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
plt.plot(p[:,0], p[:,1], 'g.')
plt.show()






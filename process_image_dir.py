#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 18:45:22 2021
@author: sariyanidi
"""

import os
import cv2
import argparse
import face_alignment_opencv as fao
import numpy as np
from glob import glob

# Parse command-line parameters
parser = argparse.ArgumentParser()
parser.add_argument("images_dir", type=str, default="samples/testdir", help="Path to the directory that contains the images to process")
parser.add_argument("--single_face", type=int, default=1, help="Set to 0 for aligning one face per image or to 1 for aligning as many faces as are detected")
parser.add_argument("--device", type=str, default='cuda', help="""Device to process. Must be set to either 'cpu' or 'cuda'. Default is 'cuda'
                    OpenCV must be compiled with CUDA and CUDNN support to really use GPU support, otherwise the software will run on CPU""")
parser.add_argument("--detection_threshold", type=float, default=0.3, help="Threshold for face detection. Default is 0.3")
parser.add_argument("--save_result_image", type=int, default=1, help="Set to 1 (Default) to save the resulting image (next to the original file) or to 0 otherwise")
parser.add_argument("--save_result_landmarks", type=int, default=1, help="Set to 1 (Default) to save resulting landmarks as .txt file or to 0 otherwise")

args = parser.parse_args()

image_paths = glob('%s/*png' % args.images_dir) + glob('%s/*jpg' % args.images_dir) + glob('%s/*bmp' % args.images_dir)

# Load face detector and aligner
detector = fao.FaceDetector(threshold=args.detection_threshold, device=args.device)
aligner = fao.FaceAligner(device=args.device)

# Create the directory to store the visual results
if args.save_result_image:
    visual_results_dir = os.path.join(args.images_dir, 'aligned_results')
    if not os.path.exists(visual_results_dir):
        os.mkdir(visual_results_dir)

# Now process each image
for image_path in image_paths:
    # read image
    im = cv2.imread(image_path)
    rim = im.copy()
    
    # detect faces
    detections = detector.get_detections(im, single_face=bool(args.single_face))
    
    # align faces
    landmark_sets = []
    for (x0, y0, xf, yf) in detections:
        p = aligner.get_landmarks(im.copy(), x0, y0, xf, yf)
        landmark_sets.append(p)
    
    # save landmarks as txt file. Each row in the output file
    # corresponds to one face's landmarks in the format x1, y1, x2, y2, ..., x68, y68
    if args.save_result_landmarks:
        num_det = len(landmark_sets)
        out = np.zeros((num_det, 68*2), dtype=int)
        
        for i in range(num_det):
            out[i,:] = landmark_sets[i].reshape(-1,)
    
        out_path = '.'.join(image_path.split('.')[0:-1])+'.txt'
        np.savetxt(out_path, out, fmt='%.2f')
        
    # the rest of the code optionally visualizes and saves the results
    if args.save_result_image or args.visualize_result:
        for p in landmark_sets:
            for ip in range(p.shape[0]):
                cv2.circle(rim, (p[ip,0], p[ip,1]), 3, (0, 255, 0), -2)
    
    # save results if asked
    if args.save_result_image:
        vis_path = os.path.join(visual_results_dir, os.path.basename(image_path))
        print('Image with detected landmarks is saved to %s' % vis_path)
        cv2.imwrite(vis_path,  rim)
        

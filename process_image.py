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

# Parse parameters
parser = argparse.ArgumentParser()
parser.add_argument("image_path", type=str, default="samples/barca.jpg", help="Path of the image to process.")
parser.add_argument("--single_face", type=int, default=0, help="Set to 0 for aligning one face per image or to 1 for aligning as many faces as are detected")
parser.add_argument("--flip_input", type=int, default=1, help="Set to 1 to flip the input while making the predictions, or to 0 to to not flip. Flipping typically leads to more robust results but reduces speed by 50%.")
parser.add_argument("--device", type=str, default='cuda', help="""Device to process. Must be set to either 'cpu' or 'cuda'. Default is 'cuda'.
                    OpenCV must be compiled with CUDA and CUDNN support to really use GPU support, otherwise the software will run on CPU.""")
parser.add_argument("--detection_threshold", type=float, default=0.3, help="Threshold for face detection. Default is 0.3.")
parser.add_argument("--visualize_result", type=int, default=1, help="Set to 1 (Default) to visualize face alignment to 0 to skip visualization.")
parser.add_argument("--save_result_image", type=int, default=1, help="Set to 1 (Default) to save the resulting image (next to the original file) or to 0 otherwise")
parser.add_argument("--save_result_landmarks", type=int, default=1, help="Set to 1 (Default) to save resulting landmarks as .txt file or to 0 otherwise")

args = parser.parse_args()

if not os.path.exists(args.image_path):
    print('Could not find file %s! Exiting.' % args.image_path)
    exit()

# Load face detector and aligner
detector = fao.FaceDetector(threshold=args.detection_threshold, device=args.device)
aligner = fao.FaceAligner(device=args.device, flip_input=args.flip_input)

# read image
im = cv2.imread(args.image_path)

# detect faces
detections = detector.get_detections(im, single_face=bool(args.single_face))

# align faces
landmark_sets = []
for (x0, y0, xf, yf) in detections:
    p = aligner.get_landmarks(im, x0, y0, xf, yf)
    landmark_sets.append(p)

# save landmarks as txt file. Each row in the output file
# corresponds to one face's landmarks in the format x1, y1, x2, y2, ..., x68, y68
if args.save_result_landmarks:
    num_det = len(landmark_sets)
    out = np.zeros((num_det, 68*2), dtype=int)
    
    for i in range(num_det):
        out[i,:] = landmark_sets[i].reshape(-1,)

    out_path = '.'.join(args.image_path.split('.')[0:-1])+'.txt'
    np.savetxt(out_path, out, fmt='%.2f')
    
# the rest of the code optionally visualizes and saves the results
if args.save_result_image or args.visualize_result:
    for p in landmark_sets:
        for ip in range(p.shape[0]):
            cv2.circle(im, (p[ip,0], p[ip,1]), 3, (0, 255, 0), -2)

if args.save_result_image:
    vis_path = '.'.join(args.image_path.split('.')[0:-1])+'_aligned.png'
    print('Image with detected landmarks is saved to %s' % vis_path)
    cv2.imwrite(vis_path,  im)
    
if args.visualize_result:    
    cv2.imshow("Aligned Face(s)", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


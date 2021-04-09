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
import pandas as pd


# Parse parameters
parser = argparse.ArgumentParser()
parser.add_argument("video_path", type=str, default="./samples/la_boheme.mp4", help="Path of the image to process.")
parser.add_argument("--single_face", type=int, default=1, help="Set to 0 for aligning one face per image or to 1 for aligning as many faces as are detected")
parser.add_argument("--device", type=str, default='cuda', help="""Device to process. Must be set to either 'cpu' or 'cuda'. Default is 'cuda'.
                    OpenCV must be compiled with CUDA and CUDNN support to really use GPU support, otherwise the software will run on CPU.""")
parser.add_argument("--detection_threshold", type=float, default=0.3, help="Threshold for face detection. Default is 0.3.")
parser.add_argument("--visualize_result", type=int, default=1, help="Set to 1 (Default) to visualize face alignment to 0 to skip visualization.")
parser.add_argument("--save_result_video", type=int, default=1, help="Set to 1 (Default) to save the resulting image (next to the original file) or to 0 otherwise")
parser.add_argument("--save_result_landmarks", type=int, default=1, help="Set to 1 (Default) to save resulting landmarks as .txt file or to 0 otherwise")

args = parser.parse_args()

if not os.path.exists(args.video_path):
    print('Could not find file %s! Exiting.' % args.video_path)
    exit()


if args.save_result_landmarks:
    df = pd.DataFrame()
    
    data_out_keys = ['face_id']
    data = {'face_id': []}
    for i in range(68):
        data_out_keys.append('x%d'%i)
        data_out_keys.append('y%d'%i)
        data['x%d'%i] = []
        data['y%d'%i] = []
        

# Load face detector and aligner
detector = fao.FaceDetector(threshold=args.detection_threshold, device=args.device)
aligner = fao.FaceAligner(device=args.device)

cap = cv2.VideoCapture(args.video_path)

if args.save_result_video:
    out_vidpath = '.'.join(args.video_path.split('.')[0:-1])+'_aligned.avi'
    
    cap_result = cv2.VideoWriter(out_vidpath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), cap.get(cv2.CAP_PROP_FPS), 
                                 (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

frame_idx = 0
while(True):    
    print('\rProcessing frame %d/%d'%(frame_idx, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), end="")
    frame_idx += 1
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break 

    # detect faces
    detections = detector.get_detections(frame, single_face=bool(args.single_face))

    # align faces
    landmark_sets = []
    for idet, (x0, y0, xf, yf) in enumerate(detections):
        p = aligner.get_landmarks(frame, x0, y0, xf, yf)
        landmark_sets.append(p)
        
        if args.save_result_landmarks:
            data['face_id'].append(idet)
        
        for ip in range(68):
            if args.save_result_landmarks:
                data['x%d'%ip].append(p[ip,0])
                data['y%d'%ip].append(p[ip,1])
            
            if args.visualize_result or args.save_result_video:
                cv2.circle(frame, (p[ip,0], p[ip,1]), 3, (0, 255, 0), -2)
    
    if args.visualize_result:
        cap_result.write(frame)
print('\n')
    
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

if args.save_result_video:
    cap_result.release()
    print('Saved result video to %s' % out_vidpath)

# save landmarks as txt file. Each row in the output file
# corresponds to one face's landmarks in the format x1, y1, x2, y2, ..., x68, y68
if args.save_result_landmarks:
    csv_path = '.'.join(args.video_path.split('.')[0:-1])+'.csv'

    df = pd.DataFrame(data, columns=data_out_keys)
    df.to_csv(csv_path, index=False)
    
    print('Saved facial landmarks to %s' % csv_path)



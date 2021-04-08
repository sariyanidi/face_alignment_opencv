#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 16:46:16 2021

@author: v
"""

import numpy as np
import cv2
import os

from requests import get  # to make GET request

def download(url, file_name):
    # open in binary mode
    with open(file_name, "wb") as file:
        # get request
        response = get(url)
        # write to file
        file.write(response.content)

class FaceDetector:
    
    def __init__(self):
        prototxt_url = "http://www.sariyanidi.com/media/deploy.prototxt"
        model_url = "http://www.sariyanidi.com/media/res10_300x300_ssd_iter_140000_fp16.caffemodel"

        module_dir = os.path.join(os.path.dirname(__file__), 'models')
        if not os.path.exists(module_dir):
            os.mkdir(module_dir)
        
        net_prototxt_fname = 'deploy.prototxt'
        net_model_fname = 'res10_300x300_ssd_iter_140000_fp16.caffemodel'
        
        self.net_prototxt_fpath = os.path.join(module_dir, net_prototxt_fname)
        self.net_model_fpath = os.path.join(module_dir, net_model_fname)
        
        if not os.path.exists(self.net_prototxt_fpath):
            print('Downloading face detection model...')
            download(prototxt_url, self.net_prototxt_fpath)
        
        if not os.path.exists(self.net_model_fpath):
            download(model_url, self.net_model_fpath)
            print('Done.')
        
        self.net = cv2.dnn.readNetFromCaffe(self.net_prototxt_fpath, self.net_model_fpath)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def get_single_detection(self, im):
        (h, w) = im.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(im, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
         
        self.net.setInput(blob)
        detections = self.net.forward()
        
        x0 = None
        y0 = None
        xf = None
        yf = None
        
        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]
            
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence < 0.5:
                continue
            
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x0, y0, xf, yf) = box.astype("int")
            break
        
        return (x0, y0, xf, yf)
    

        

class FaceAligner:
    
    def __init__(self):
        model_url = "http://www.sariyanidi.com/media/model_FAN_frozen.pb"

        module_dir = os.path.join(os.path.dirname(__file__), 'models')
        if not os.path.exists(module_dir):
            os.mkdir(module_dir)
        
        net_model_fname = 'model_FAN_frozen.pb'
        
        self.net_model_fpath = os.path.join(module_dir, net_model_fname)
        
        if not os.path.exists(self.net_model_fpath):
            print('Downloading face alignment model...')
            download(model_url, self.net_model_fpath)
            print('Done.')
        
        self.net = cv2.dnn.readNetFromTensorflow(self.net_model_fpath)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def get_landmarks(self, im, x0, y0, xf, yf):
        (imp, ptCenter, scale) = preprocess_image(im, x0, y0, xf, yf)
        return get_landmarks(imp, self.net, ptCenter, scale)

def recolor_image(image, c1, c2, c3):
    image[:,:,0] = c1*image[:,:,0]
    image[:,:,1] = c2*image[:,:,1]
    image[:,:,2] = c3*image[:,:,2]
    
    image[image>1.0] = 1.0
    image[image<0.0] = 0.0

    return image

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def scale_image(image, s):
    image_center = tuple(np.array(image.shape[1::-1], dtype=np.float32) / 2)
    image_center_after = tuple(s*np.array(image.shape[1::-1], dtype=np.float32) / 2)
    image_shift = (image_center[0]-image_center_after[0],
                 image_center[1]-image_center_after[1])
  
    M = np.float32([[s,0,image_shift[0]],[0,s,image_shift[1]]])
  
    result = cv2.warpAffine(image, M, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def perturb_image(image, s, angle):
    return scale_image(rotate_image(image, angle), s)




def transform(point, center, scale, resolution, invert=False):
    """Generate and affine transformation matrix.

    Given a set of points, a center, a scale and a targer resolution, the
    function generates and affine transformation matrix. If invert is ``True``
    it will produce the inverse transformation.

    Arguments:
        point {torch.tensor} -- the input 2D point
        center {torch.tensor or numpy.array} -- the center around which to perform the transformations
        scale {float} -- the scale of the face/object
        resolution {float} -- the output resolution

    Keyword Arguments:
        invert {bool} -- define wherever the function should produce the direct or the
        inverse transformation matrix (default: {False})
    """
    _pt = np.ones(3)
    _pt[0] = point[0]
    _pt[1] = point[1]

    h = 200.0 * scale
    t = np.eye(3)
    t[0, 0] = resolution / h
    t[1, 1] = resolution / h
    t[0, 2] = resolution * (-center[0] / h + 0.5)
    t[1, 2] = resolution * (-center[1] / h + 0.5)
    
    if invert:
        t = np.linalg.inv(t)

    new_point = (np.matmul(t, _pt))[0:2]
    

    return np.array(new_point, dtype=int)#.int()


def crop(image, center, scale, resolution=256.0):
    """Center crops an image or set of heatmaps

    Arguments:
        image {numpy.array} -- an rgb image
        center {numpy.array} -- the center of the object, usually the same as of the bounding box
        scale {float} -- scale of the face

    Keyword Arguments:
        resolution {float} -- the size of the output cropped image (default: {256.0})

    Returns:
        [type] -- [description]
    """  # Crop around the center point
    """ Crops the image around the center. Input is expected to be an np.ndarray """
    ul = transform([1, 1], center, scale, resolution, True)
    br = transform([resolution, resolution], center, scale, resolution, True)
    # pad = math.ceil(torch.norm((ul - br).float()) / 2.0 - (br[0] - ul[0]) / 2.0)
    if image.ndim > 2:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0],
                           image.shape[2]], dtype=np.int32)
        newImg = np.zeros(newDim, dtype=np.uint8)
    else:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0]], dtype=np.int)
        newImg = np.zeros(newDim, dtype=np.uint8)
        
    ht = image.shape[0]
    wd = image.shape[1]
    newX = np.array(
        [max(1, -ul[0] + 1), min(br[0], wd) - ul[0]], dtype=np.int32)
    newY = np.array(
        [max(1, -ul[1] + 1), min(br[1], ht) - ul[1]], dtype=np.int32)
    
    oldX = np.array([max(1, ul[0] + 1), min(br[0], wd)], dtype=np.int32)
    oldY = np.array([max(1, ul[1] + 1), min(br[1], ht)], dtype=np.int32)
    newImg[newY[0] - 1:newY[1], newX[0] - 1:newX[1]
           ] = image[oldY[0] - 1:oldY[1], oldX[0] - 1:oldX[1], :]
    newImg = cv2.resize(newImg, dsize=(int(resolution), int(resolution)),
                        interpolation=cv2.INTER_LINEAR)
    
    return newImg


def get_landmarks(imp, landmark_net, ptCenter, scale):
    landmark_net.setInput(cv2.dnn.blobFromImage(np.float32(cv2.cvtColor(imp, cv2.COLOR_BGR2RGB))/255))
    y_dnn = landmark_net.forward()
    
    return heatmaps_to_landmarks(y_dnn, ptCenter, scale)


def preprocess_image(im, x0, y0, xf, yf):
    face_width = xf - x0
    face_height = yf - y0
    
    ptCenter = np.array([(x0+x0+face_width)/2.0, (y0+y0+face_height)/2.0])
    scale = (face_width+face_height)/195.0
    
    return (crop(im, ptCenter, scale), ptCenter, scale)


def heatmaps_to_landmarks(hm, ptCenter, scale, resolution=64):
    p = np.zeros((68,2), dtype=int)
    
    for i in range(68):
        (_, _, _, maxLoc) = cv2.minMaxLoc(hm[0,i,:,:])
        
        px = maxLoc[0]
        py = maxLoc[1]
        
        
        
        if px > 0 and px < 63 and py > 0 and py < 63:
            diffx = hm[0,i,py,px+1] - hm[0,i,py,px-1]
            diffy = hm[0,i,py+1,px] - hm[0,i,py-1, px];
            
            px += 1
            py += 1 
            
            if diffx > 0:
                px += 0.25
            else:
                px -= 0.25
        
            if diffy > 0:
                py += 0.25
            else:
                py -= 0.25
    
        px -= 0.5
        py -= 0.5

        ptOrig = transform((px, py), ptCenter, scale, resolution, True)
        p[i,0] = ptOrig[0]
        p[i,1] = ptOrig[1]

    
    return p









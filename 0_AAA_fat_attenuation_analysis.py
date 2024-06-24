#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Clinic for Diagnositic and Interventional Radiology, University Hospital Bonn, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import sys, json, math
from tqdm import tqdm
from copy import deepcopy
import numba as nb
import pickle
import seaborn as sns
sns.set() 
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage

import pandas as pd
import numpy as np
import nibabel as nib
import os
import pathlib
import cv2
import sys
import torch

import time

from skimage.transform import resize
from skimage.segmentation import flood_fill
         
###### DATA SETTINGS
inputFolder = 'NIFTI_noWall_1mm_Nierenart'
imageFilenameAppendix = '_image'
segmFilenameAppendix = '_Segmentierung'
landmarkAppendix = 'f_'
CenterLineAppendix = '_centerLine'

nameAfterlandmarkAppendix = True #'f_BAA01'
dataTypeLandmark = '.fcsv'#'.mrk.json'
dataTypeIn = '.nii.gz'
dataTypeOut_images = '.nii'#.npy'#'.nii'
dataTypeOut_labels = '.nii.gz'#.npy'#'.nii'

group_a_appendix = 'BAA_'
group_b_appendix = 'K_'

namesLandmarks_a = ['Bifurkation',
                    'Truncus_coeliacus',
                    'unteres_Ende_BAA',
                   'oberes_Ende_BAA',
                   'RCA_Abgang']

namesLandmarks_b = ['Bifurkation',
                    'Truncus_coeliacus',
                   'Abgang_rechte_Nierenarterie',
                   'RCA_Abgang']

##### BAA BEREICH
namesSections_a = ['Rest',
                     'unteres_Ende_BAA-50(mm)',
                     'unteres_Ende_BAA-oberes_Ende_BAA',
                     'oberes_Ende_BAA+50(mm)'
                     ]
removeSections_a = None
namesSections_b = ['Rest', 'Bifurkation-Abgang_rechte_Nierenarterie']
removeSections_b = None

#####  Bifurkation-Truncus_coeliacus OHNE BAA
# namesSections_a = ['Rest',
#                     'Bifurkation-Truncus_coeliacus']
# removeSections_a = ['unteres_Ende_BAA-oberes_Ende_BAA']
# namesSections_b = ['Rest',
#                     'Bifurkation-Truncus_coeliacus']
# removeSections_b  = None

#####  Bifurkation-Abgang_rechte_Nierenarterie_coeliacus MIT BAA
# namesSections_a = ['Rest',
#                     'Bifurkation-Abgang_rechte_Nierenarterie']
# removeSections_a = None
# namesSections_b = ['Rest',
#                     'Bifurkation-Abgang_rechte_Nierenarterie']
# removeSections_b  = None

#####  Bifurkation-Abgang_rechte_Nierenarterie_coeliacus OHNE BAA
# namesSections_a = ['Rest',
#                     'Bifurkation-Abgang_rechte_Nierenarterie']
# removeSections_a = ['unteres_Ende_BAA-oberes_Ende_BAA']
# namesSections_b = ['Rest',
#                     'Bifurkation-Abgang_rechte_Nierenarterie']
# removeSections_b  = None


colorsPlot_b = ['tab:blue','tab:orange']
colorsPlot_a = ['tab:blue','tab:orange']

showErrorBands = False #default
alpha_errorBands = 0.2 #default

###### postProc SETTINGS
postProc_imClose = True 
postProc_kernelSize_imClose = (5,5,5) #default (5,5,5)
postProc_extractTwoLargest = True #default True
postProc_dilateAorticWall  = False #default True
aorticWallThickenss = None#1 #default 1 #mm
postProc_getCenterLine = True #default True
postProc_getCenterLine_res = (2,2,2) #default (2,2,2)#mm
postProc_getCenterLine_kernelSizeInMM = (80,80,10) #default (80,80,10)#mm

###### KALK SETTINGS
kalkThresh = 400  #laut marillia 400 aber klappt schon direktbei 1250 garnicht!
#if more than X% of volume is within base thresh 
#increase the thresh to a value that only X% of volume if wihtin
adaptiveKalkThresh = True #default True
percOfVolumeInThresh = 2.5 #default 2.5
imCloseKalkMask = True #default True
kernelSize_imCloseKalkMask = (3,3,3) #default (3,3,3)#mm

###### FAT SETTINGS
fatWindow = (-190, -30) #default
# rangesFromAorticWall = np.arange(1.5,21,1.5) #default
numVoxelOneStep = 1
maxRangeFromAorta = 20#mm
smallesVoxelspacingToExpect = 0.5#mm
imCloseFatMask = False #default False
kernelSize_imCloseFatMask = (3,3,3) #default (3,3,3)

###### UTILS SETTINGS
overWriteStages = False
writeFiles = True
plotLandmarksPng = True

#0.8 s with numba
#10.1 s without numba
@nb.njit(parallel=False, fastmath=False)
def partitionAtPlanes( segm, indicies_segm, midPoints, normVectors, old_vals_wall, new_vals_wall, cutLower=False ):
    #asign segm coordiantes new classes
    for segm_point in indicies_segm: #check every point
        for j, midPoint in enumerate(midPoints): #go threw landmarks
            # if the segm_point-landmark-vector starts to not be in same direction
            # as the normVectors you found the section
            if np.dot( segm_point - midPoint , normVectors[j] ) < 0:
                #dont take the first segm under first ladnmark (cut the bottom)
                if j>0:
                    val = segm[segm_point[0],segm_point[1],segm_point[2]]
                    if val in new_vals_wall: 
                        val = old_vals_wall[new_vals_wall.index(val)]
                    segm[segm_point[0],segm_point[1],segm_point[2]] = len(midPoints)*(val-1)+j+1
                else:
                    if cutLower: segm[segm_point[0],segm_point[1],segm_point[2]] = 0 #cut!
                break #dont look further ladnmarks as they will be also in diff direction
    return segm

def partitionAtPlanes_debug( segm, indicies_segm, midPoints, normVectors, old_vals_wall, new_vals_wall, cutLower=False ):
    #asign segm coordiantes new classes
    for segm_point in indicies_segm: #check every point
        for j, midPoint in enumerate(midPoints): #go threw landmarks
            # if the segm_point-landmark-vector starts to not be in same direction
            # as the normVectors you found the section
            if np.dot( segm_point - midPoint , normVectors[j] ) < 0:
                #dont take the first segm under first ladnmark (cut the bottom)
                if j>0:
                    val = segm[segm_point[0],segm_point[1],segm_point[2]]
                    if val in new_vals_wall: 
                        val = old_vals_wall[new_vals_wall.index(val)]
                    segm[segm_point[0],segm_point[1],segm_point[2]] = len(midPoints)*(val-1)+j+1
                else:
                    if cutLower: segm[segm_point[0],segm_point[1],segm_point[2]] = 0 #cut!
                break #dont look further ladnmarks as they will be also in diff direction
    return segm
            
@nb.njit(parallel=False, fastmath=False)
def get_meanDistance(cur_indicies_wall, idx ):

    curMeanDistance = 0
    for wall_idx in cur_indicies_wall:
        wall_idx_inOrig = wall_idx
        # curMeanDistance += np.linalg.norm(idx-wall_idx_inOrig) 
        dist_vec = idx-wall_idx_inOrig
        curMeanDistance += math.sqrt( dist_vec[0]**2+dist_vec[1]**2+dist_vec[2]**2)**2
    return curMeanDistance / len(cur_indicies_wall)

@nb.njit(parallel=False, fastmath=False) 
def tranformPoints_w_ijk( points_w, affine ):
    affine_inv = np.linalg.inv(affine)
    points_ijk = np.zeros(np.shape(points_w))
    points_w[:,0] *= -1
    points_w[:,1] *= -1
    for idx in range(len(points_w)):
        points_ijk[idx] = affine_inv.dot(points_w[idx])
    return points_ijk

@nb.njit(parallel=False, fastmath=False) 
def tranformPoints_ijk_w( points_ijk, affine ):
    points_w = np.zeros(np.shape(points_ijk))
    for idx in range(len(points_ijk)):
        curPos =  affine.dot(points_ijk[idx])
        curPos[0] *= -1
        curPos[1] *= -1
        points_w[idx] = curPos
    return points_w

def getIdxLandmarksOfSections(namesSections, namesLandmarks): 
    idxSections = []
    for nameSection in namesSections:
        for landmark in namesLandmarks :
            if landmark in nameSection:
                idxSections.append( namesLandmarks.index(landmark) )
    idxSections.sort()  
    return idxSections

def get_normVectors_by_centerLineIdx(centerLineCords_ijk, idx_closestCenterPos):
    normVectors =  [
       unit_vector(( (centerLineCords_ijk[idx+1]-centerLineCords_ijk[idx]) + \
                     (centerLineCords_ijk[idx]-centerLineCords_ijk[idx-1]) )/2) \
       if idx<len(centerLineCords_ijk)-2 and idx>1 \
       else \
       unit_vector(( (centerLineCords_ijk[idx]-centerLineCords_ijk[idx-1]) + \
                     (centerLineCords_ijk[idx-1]-centerLineCords_ijk[idx-2]) )/2) \
       if idx>1 \
       else \
       unit_vector(( (centerLineCords_ijk[idx+2]-centerLineCords_ijk[idx+1]) + \
                    (centerLineCords_ijk[idx+1]-centerLineCords_ijk[idx]) )/2) \
           for idx in idx_closestCenterPos ]  
    return normVectors
                    

def createDictFromKeysAndVals(keys, vals):
    vals = [ [x] for x in vals]
    return dict(zip(keys,vals))
    
def appendToDictFromKeysAndVals(curDict, keys, vals):
    for key,val in zip(keys,vals):
        curDict[key] += [val]
    return curDict

def checkifValueHasDecimals( val ):
    val_orig = np.copy(val)
    if int(val) == val_orig: return False
    else: return True
                
def getNormOfPlaneByPoints( points ):
    # https://kitchingroup.cheme.cmu.edu/blog/2015/01/18/Equation-of-a-plane-through-three-points/
    p1, p2, p3 = points
    # These two vectors are in the plane
    v1 = p3 - p1
    v2 = p2 - p1
    # the cross product is a vector normal to the plane
    return unit_vector(np.cross(v1, v2))
    
def postProc_imCloseCoronar(mask):
    mask = mask.astype(np.uint8)
    kernel_size = 3
    kernel = np.uint8(np.zeros((kernel_size,kernel_size)))
    kernel[int(np.floor(kernel_size/2)),:] = 1
    for z in range(np.shape(mask)[1]):
        mask[:,z,:] = cv2.morphologyEx(mask[:,z,:], cv2.MORPH_CLOSE, kernel)
    return mask

def postProc_extractLargestBlob(mask):
    mask = mask.astype(np.uint8)
    labelMap, num_objects = ndimage.label(mask) 
    sizes = ndimage.sum(mask, labelMap, range(num_objects + 1))
    mask = labelMap == np.argmax(sizes, axis=0)
    return mask

def postProc_extractTwoLargestBlobs(mask):
    mask = mask.astype(np.uint8)
    labelMap, num_objects = ndimage.label(mask) 
    if num_objects > 1:
        sizes = ndimage.sum(mask, labelMap, range(num_objects + 1))
        argmax_list = np.argsort(sizes)
        mask = labelMap == argmax_list[-1]
        mask += labelMap == argmax_list[-2]
    return mask

def norm_transformInRange01(x,  bgThresh = None, makeImMinZeroPriorNorm=False):
    min_val = np.min(x)
    max_val = np.max(x)
    if max_val-min_val != 0:
        x = (x-min_val)/(max_val-min_val)
    return x
   
def norm_transformInRange0_95Perc(x, bgThresh = None, makeImMinZeroPriorNorm = True):       
    if makeImMinZeroPriorNorm:
        min_im = np.min(x)
        if min_im < 0:
            x -= min_im
            min_im = 0
        
    if bgThresh is not None:
        if np.max(x) <= bgThresh:
            return x # nothing in image to norm
        mask = x>bgThresh
        upper_percentile  = np.percentile(x[mask],95)
    else:
        upper_percentile  = np.percentile(x,95)
    max_val = upper_percentile
    min_val = min_im
    if max_val-min_val != 0:
        x = (x-min_val)/(max_val-min_val)
    return x

def norm_transformZeroMeanUnitVar(x, bgThresh = 0, makeImMinZeroPriorNorm = True): 
    if makeImMinZeroPriorNorm:
        min_im = np.min(x)
        if min_im < 0:
            x -= min_im
            min_im = 0
    
    if bgThresh is None:
        mean_im = np.mean(x)
        std_im = np.std(x)  
    else:
        mask = x>bgThresh
        mean_im = np.mean(x[mask])
        std_im = np.std(x[mask])       

    x -= mean_im
    if std_im != 0:
        x /= std_im  
    return x

def plotMultiView(img, planes= ['axial', 'coronar','saggital'], savePath=None, centerCoord=None, extensionFilename='_orig'):
    
    for plane in planes:
        
        if plane== 'axial': image_plot = img[:,:,centerCoord[2]]  
        elif plane == 'saggital': image_plot = img[centerCoord[0],:,:]  
        else: image_plot = img[:,centerCoord[1],:]
            
        if not plane == 'axial': image_plot = np.flip(image_plot)  
  
        fig, ax = plt.subplots(1,1)
        ax.imshow(image_plot).set_cmap('gray')
        ax.axis('off')
        savePath = savePath.replace('.png', extensionFilename+'.png')
        Path( os.path.dirname(savePath)).mkdir(parents=True, exist_ok=True)
        fig.savefig(savePath)  
        plt.close(fig)    
        
def getLargestContour(mask, nLargest=1):
    #https://stackoverflow.com/questions/56589691/how-to-leave-only-the-largest-blob-in-an-image
    # Find largest contour in intermediate image
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    out = np.zeros(mask.shape, np.uint8)
    for n in range(nLargest):
        if n > len(cnts_sorted)-1:
            break
        # find maximum area contour
        cv2.drawContours(out, cnts_sorted, n, 1, cv2.FILLED) #[cnts_sorted[n]]
        
    return cv2.bitwise_and(mask, out)

def unit_vector(vector):
    """ Returns the unit vector of the vector.  https://stackoverflow.com/a/13849249 """
    return vector / np.linalg.norm(vector)

def ball_kernel(r):
    sideLenght = 2*r+1
    mid_pos = np.asarray([r,r,r])
    kernel = np.ones( (sideLenght, sideLenght, sideLenght), dtype=int )
    for i in range(sideLenght):
        for j in range(sideLenght):
            for k in range(sideLenght):
                pos = np.asarray([i,j,k])
                dist = np.linalg.norm(pos - mid_pos) 
                if dist > r: kernel[i,j,k] = 0
    return kernel

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def tensorDilation3D( mat, kernel):
    padding = np.array(kernel.shape)/2
    padding_asym = []
    for pad in reversed(padding): #ConstantPad3d the first say how to pad last dim dont know why
        if checkifValueHasDecimals( pad ):
            padding_asym += 2 * [int(np.floor(pad))]
        else:
            padding_asym += [int(pad)]
            padding_asym += [int(pad-1)]
    padding_asym = tuple(padding_asym)    

    pad_layer = torch.nn.ConstantPad3d(padding_asym,0)
    # https://stackoverflow.com/questions/56235733/is-there-a-tensor-operation-or-function-in-pytorch-that-works-like-cv2-dilate
    mat_tensor = pad_layer(torch.Tensor(mat).cuda()).unsqueeze(0).unsqueeze(0)
    kernel_tensor = torch.Tensor(np.expand_dims(np.expand_dims(kernel, 0), 0)).cuda() 
    mat_tensor = torch.clamp(torch.nn.functional.conv3d(mat_tensor, 
                                   kernel_tensor),
                                   0, 1)
    mat = mat_tensor.cpu().numpy()[0,0,:].astype(bool)
    mat_tensor = None
    kernel_tensor = None
    return mat

def tensorErosion3D( mat, kernel):
    padding = np.array(kernel.shape)/2
    padding_asym = []
    for pad in reversed(padding): #ConstantPad3d the first say how to pad last dim dont know why
        if checkifValueHasDecimals( pad ):
            padding_asym += 2 * [int(np.floor(pad))]
        else:
            padding_asym += [int(pad)]
            padding_asym += [int(pad-1)]
    padding_asym = tuple(padding_asym)    

    thresh = kernel.sum()
    pad_layer = torch.nn.ConstantPad3d(padding_asym,0)
    # https://stackoverflow.com/questions/56235733/is-there-a-tensor-operation-or-function-in-pytorch-that-works-like-cv2-dilate
    mat_tensor = pad_layer(torch.Tensor(mat).cuda()).unsqueeze(0).unsqueeze(0)
    kernel_tensor = torch.Tensor(np.expand_dims(np.expand_dims(kernel, 0), 0)).cuda() 
    mat_tensor = torch.nn.functional.conv3d(mat_tensor, kernel_tensor)
    mat = mat_tensor.cpu().numpy()[0,0,:]
    mat = mat == thresh
    mat_tensor = None
    kernel_tensor = None
    return mat

def tensorClose3D( mat, kernel):
    padding = np.array(kernel.shape)/2
    padding_asym = []
    for pad in reversed(padding): #ConstantPad3d the first say how to pad last dim dont know why
        if checkifValueHasDecimals( pad ):
            padding_asym += 2 * [int(np.floor(pad))]
        else:
            padding_asym += [int(pad)]
            padding_asym += [int(pad-1)]
    padding_asym = tuple(padding_asym)    

    thresh = kernel.sum()
    pad_layer = torch.nn.ConstantPad3d(padding_asym,0)
    
    mat_tensor = pad_layer(torch.Tensor(mat).cuda()).unsqueeze(0).unsqueeze(0)
    kernel_tensor = torch.Tensor(np.expand_dims(np.expand_dims(kernel, 0), 0)).cuda() 
    mat_tensor = torch.clamp(torch.nn.functional.conv3d(mat_tensor, 
                                   kernel_tensor),
                                   0, 1)
    mat_tensor = pad_layer(mat_tensor.squeeze()).unsqueeze(0).unsqueeze(0)
    kernel_tensor = torch.Tensor(np.expand_dims(np.expand_dims(kernel, 0), 0)).cuda() 
    mat_tensor = torch.nn.functional.conv3d(mat_tensor, kernel_tensor)
    mat = mat_tensor.cpu().numpy()[0,0,:]
    mat = mat == thresh
    mat_tensor = None
    kernel_tensor = None
    return mat

def tensorOpen3D( mat, kernel):
    padding = np.array(kernel.shape)/2
    padding_asym = []
    for pad in reversed(padding): #ConstantPad3d the first say how to pad last dim dont know why
        if checkifValueHasDecimals( pad ):
            padding_asym += 2 * [int(np.floor(pad))]
        else:
            padding_asym += [int(pad)]
            padding_asym += [int(pad-1)]
    padding_asym = tuple(padding_asym)    

    thresh = kernel.sum()
    pad_layer = torch.nn.ConstantPad3d(padding_asym,0)
    
    mat_tensor = pad_layer(torch.Tensor(mat).cuda()).unsqueeze(0).unsqueeze(0)
    kernel_tensor = torch.Tensor(np.expand_dims(np.expand_dims(kernel, 0), 0)).cuda() 
    mat_tensor = torch.nn.functional.conv3d(mat_tensor, kernel_tensor)
    
    mat_tensor = pad_layer(mat_tensor.squeeze()).unsqueeze(0).unsqueeze(0)
    kernel_tensor = torch.Tensor(np.expand_dims(np.expand_dims(kernel, 0), 0)).cuda() 
    mat_tensor = torch.clamp(torch.nn.functional.conv3d(mat_tensor, 
                                   kernel_tensor),
                                   0, 1)
    mat = mat_tensor.cpu().numpy()[0,0,:]
    mat = mat == thresh
    mat_tensor = None
    kernel_tensor = None
    return mat

basePath = os.path.dirname(os.path.realpath(__file__))+os.sep
inputPath = basePath+inputFolder+os.sep
postProcStagesPath = basePath+'postProcStages'+os.sep

Path(postProcStagesPath).mkdir(parents=True, exist_ok=True)
stage0_pkl = postProcStagesPath+os.sep+'stage0.pkl' 
stage1_pkl = postProcStagesPath+os.sep+'stage1.pkl' 
stage2_pkl = postProcStagesPath+os.sep+'stage2.pkl' 

# allNiiFiles = get_nii_files(inputPath)
idxSections_a = getIdxLandmarksOfSections(namesSections_a, namesLandmarks_a)
idxSections_b = getIdxLandmarksOfSections(namesSections_b, namesLandmarks_b)

idxRemoveSections_a = None
idxRemoveSections_b = None
if removeSections_a is not None:
    idxRemoveSections_a = getIdxLandmarksOfSections(removeSections_a, namesLandmarks_a)
if removeSections_b is not None:
    idxRemoveSections_b = getIdxLandmarksOfSections(removeSections_b, namesLandmarks_b)

allLandmarkFiles = []
for x in Path(inputPath).rglob('*'+dataTypeLandmark): allLandmarkFiles.append(str(x))
allLandmarkFiles = [x for x in allLandmarkFiles if not CenterLineAppendix in x]

filenames = []
pathsSegm = [] 
pathsIm = [] 
pathsLandmarks = [] 

voxelSpacings = [] 
cutCoordZ = [] 
allOrigImageFiles = []
matrixSizes =  []
allPositions_ijk =  [] 
allPositions_w =  [] 
allPositions_ijk_cutZ =  [] 
allPositions_w_cutZ =  []

kalkThreshUsed =  []

#postproc if not already did
for path in tqdm(allLandmarkFiles):
    
    # -------- loading START -------- 
    curPathLandmark = path
    curPathFolder = os.path.dirname(path)+os.sep
    curFolderName = curPathFolder.split(os.sep)[-2]
    curFilename = curFolderName
    
    curPathSegm = curPathFolder+curFilename+segmFilenameAppendix+dataTypeIn
    curPathSegm_postProc = curPathFolder+curFilename+segmFilenameAppendix+'_postProc'+dataTypeIn
    curPathIm = curPathFolder+curFilename+imageFilenameAppendix+dataTypeIn
    
    curCenterLineCsv = curPathFolder+curFilename+CenterLineAppendix+'.fcsv'

    if not os.path.isfile(curPathSegm_postProc):
        niiFile = nib.load(curPathIm)
        curVoxelSpacing = niiFile.header.get_zooms()
        im_shape = niiFile.header.get_data_shape()
        
        segm = nib.load(curPathSegm).get_fdata() > 0
        
        assert im_shape == segm.shape, f'image {curPathIm} and semg {curPathSegm} dont have same shape ! Export error from slicer !'
            
        # segm = ndimage.binary_closing(segm, structure=np.ones((5,5,5), dtype=bool) )
        if postProc_imClose: segm = tensorClose3D(segm, np.ones(postProc_kernelSize_imClose)).astype(bool)
        if postProc_extractTwoLargest: segm = postProc_extractTwoLargestBlobs(segm).astype(np.uint8)
        #inlcude aorta wall into segm for FAI  
        
        if postProc_getCenterLine:
            segmentSize = (np.asarray(postProc_getCenterLine_kernelSizeInMM)/2).astype(int)
            postProc_getCenterLine_res = np.asarray(postProc_getCenterLine_res)
            scale_factor = curVoxelSpacing/postProc_getCenterLine_res
            new_shape = np.round(np.asarray(segm.shape)*scale_factor).astype(int)
            segm_lowRes = resize( np.copy(segm), new_shape,
                                      order=0, anti_aliasing=False, mode='constant', cval=0).astype(bool)
            kernel = np.array([  [[0,0,0],
                                  [0,1,0],
                                  [0,0,0]],
                                 [[0,1,0],
                                  [1,1,1],
                                  [0,1,0]],
                                 [[0,0,0],
                                  [0,1,0],
                                  [0,0,0]] ]).astype(int) 
            segm_wall = tensorDilation3D( np.copy(segm_lowRes), kernel)
            segm_wall[ segm_lowRes ] = False
            
            indicies_segm = np.zeros( (np.sum(segm_lowRes), 3))
            indicies_segm[:,0], indicies_segm[:,1], indicies_segm[:,2] = np.where( segm_lowRes)
            indicies_segm = indicies_segm.astype(int)
  
            new_shape = np.shape(segm_wall)
            new_shape_min = np.min(new_shape)
            distanceMap = np.zeros(new_shape)
            
            for idx in tqdm(indicies_segm):
                
                curWallSegmentBorders = np.asarray([idx-segmentSize/2,
                                         idx+segmentSize/2]).astype(int)
                if np.any(curWallSegmentBorders < 0):
                    curWallSegmentBorders[curWallSegmentBorders < 0] = 0
                
                if np.any(curWallSegmentBorders > new_shape_min):
                    for border_idx in range(len(curWallSegmentBorders)):
                        for dim_idx in range(len(curWallSegmentBorders[border_idx])):
                            if curWallSegmentBorders[border_idx][dim_idx] > new_shape[dim_idx]:
                                curWallSegmentBorders[border_idx][dim_idx] = new_shape[dim_idx]-1
          
                curWallSegment = segm_wall[curWallSegmentBorders[0][0]:curWallSegmentBorders[1][0],
                                                 curWallSegmentBorders[0][1]:curWallSegmentBorders[1][1],
                                                 curWallSegmentBorders[0][2]:curWallSegmentBorders[1][2]]
                
                cur_indicies_wall = np.zeros( (np.sum(curWallSegment), 3))
                cur_indicies_wall[:,0], cur_indicies_wall[:,1], cur_indicies_wall[:,2] = np.where( curWallSegment)
                cur_indicies_wall = cur_indicies_wall.astype(int)
                
                distanceMap[idx[0],idx[1],idx[2]] = get_meanDistance(cur_indicies_wall, idx-curWallSegmentBorders[0])

            z = np.arange(indicies_segm[:,2].min(),indicies_segm[:,2].max())
            centerLineCords = np.zeros( (len(z),3) )
            for j, cur_z in enumerate(z):
                curSlice = distanceMap[:,:,cur_z]
                curSlice[curSlice==0] = np.inf
                centerLineCords[j,0], centerLineCords[j,1] = \
                    np.unravel_index(curSlice.argmin(), curSlice.shape)
                centerLineCords[j,2] = cur_z
            
            #running mean over 4
            centerLineCords_rMean = np.swapaxes(np.asarray( [
                np.convolve(centerLineCords[:,0], np.ones(4)/4, mode='valid'),
                np.convolve(centerLineCords[:,1], np.ones(4)/4, mode='valid'),
                np.convolve(centerLineCords[:,2], np.ones(4)/4, mode='valid') ] ),0,1)
            
            centerLineCords_rMean /= scale_factor
            
            centerLineCords_w = tranformPoints_ijk_w( np.concatenate( 
                (centerLineCords_rMean, np.ones((len(centerLineCords_rMean),1))), axis=1),
                niiFile.affine)
            centerLineCords_w = centerLineCords_w[:,:3]
                   
            dict_centerline = {'x':centerLineCords_w[:,0],
                               'y':centerLineCords_w[:,1],
                               'z':centerLineCords_w[:,2],
                               'ow':np.zeros(len(centerLineCords_w), dtype=int),
                               'ox':np.zeros(len(centerLineCords_w), dtype=int),
                               'oy':np.zeros(len(centerLineCords_w), dtype=int),
                               'oz':np.ones(len(centerLineCords_w), dtype=int),
                               'vis':np.ones(len(centerLineCords_w), dtype=int),
                               'sel':np.ones(len(centerLineCords_w), dtype=int),
                               'lock':np.ones(len(centerLineCords_w), dtype=int),
                               'label': [ 'C-'+str(x) for x in np.arange(0,len(centerLineCords_w)) ],
                               'associatedNodeID': ['vtkMRMLScalarVolumeNode1']*len(centerLineCords_w)
                               }  
            
            df_centerline = pd.DataFrame(data=dict_centerline)
            with open(curCenterLineCsv, 'w') as text_file:
                print('# Markups fiducial file version = 4.11\n# CoordinateSystem = LPS\n# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID',
                      file=text_file)
            df_centerline.to_csv(curCenterLineCsv, mode='a', header=False)

        if postProc_dilateAorticWall and aorticWallThickenss is not None: 
            kernel = ball_kernel(int(np.round(aorticWallThickenss/curVoxelSpacing[0])))
            segm = tensorDilation3D( segm, kernel)  
            
        nib.save(nib.Nifti1Image(segm.astype(np.uint8), niiFile.affine), curPathSegm_postProc) 

dict_out_a = None
dict_out_b = None
if not os.path.isfile(stage0_pkl) or overWriteStages: 

    for i, path in enumerate(tqdm(allLandmarkFiles)):    

        # -------- loading START -------- 
        curPathLandmark = path
        curPathFolder = os.path.dirname(path)+os.sep
        curFolderName = curPathFolder.split(os.sep)[-2]
        curFilename = curFolderName
        
        curPathSegm = curPathFolder+curFilename+segmFilenameAppendix+dataTypeIn
        curPathSegm_postProc = curPathFolder+curFilename+segmFilenameAppendix+'_postProc'+dataTypeIn
       
        curPathSegm_out = curPathSegm.replace(segmFilenameAppendix, segmFilenameAppendix+'_sectioned')
        curPathSegm_out = curPathSegm_out.replace(dataTypeIn, dataTypeOut_labels)
        
        curPathIm = curPathFolder+curFilename+imageFilenameAppendix+dataTypeIn
        curPngPath = curPathFolder+'png'+os.sep
        
        curCenterLineCsv = curPathFolder+curFilename+CenterLineAppendix+'.fcsv'
        
        niiFile = nib.load(curPathIm)
        im = niiFile.get_fdata()
        segm = nib.load(curPathSegm_postProc).get_fdata().astype(bool)
        curVoxelSpacing = niiFile.header.get_zooms()
        curFactorNumToVol =  curVoxelSpacing[0]*curVoxelSpacing[1]*curVoxelSpacing[2]
        
        filenames.append(curFilename)
        pathsSegm.append(curPathSegm)
        pathsIm.append(curPathIm)
        pathsLandmarks.append(curPathLandmark)
        voxelSpacings.append(curVoxelSpacing)
        im_shape = np.shape(im)
        matrixSizes.append(im_shape)
        
        #only take positions of interest, that represent sections to analyse
        if group_a_appendix in curFilename: 
            idxSections = idxSections_a
            idxRemoveSections = idxRemoveSections_a
            namesLandmarks = namesLandmarks_a
            namesSections = namesSections_a
            colorsPlot = colorsPlot_a
        elif group_b_appendix in curFilename: 
            idxSections = idxSections_b
            idxRemoveSections = idxRemoveSections_b
            namesLandmarks = namesLandmarks_b
            namesSections = namesSections_b
            colorsPlot = colorsPlot_b
        else:
            print(f'Could not group {curFilename} to {group_a_appendix} or {group_b_appendix}. Ignoring ...')
            continue
        
        numSteps_total = int(np.ceil(maxRangeFromAorta/(smallesVoxelspacingToExpect*numVoxelOneStep)))
        numStepsFromAorticWall = int(np.round(maxRangeFromAorta/curVoxelSpacing[0]/numVoxelOneStep))
        if numStepsFromAorticWall+1 > numSteps_total:
            numStepsFromAorticWall = numSteps_total-1
        IndiciesStepsFromAorticWall = np.arange(numStepsFromAorticWall)+1    
            
        if not os.path.isfile(curPathSegm_out) or overWriteStages:
            # -------- centerline loading START --------
            centerLineCords_w = []
            df = pd.read_csv(curCenterLineCsv, header=None,  skiprows=[0,1,2])
            for index, row in df.iterrows():
                centerLineCords_w.append([row[1], row[2], row[3]])
            df = None
    
            centerLineCords_ijk = tranformPoints_w_ijk( np.concatenate( 
                (centerLineCords_w, np.ones((len(centerLineCords_w),1))), axis=1),
                niiFile.affine)
            centerLineCords_ijk = centerLineCords_ijk[:,:3]
    
            # -------- loading END -------- 
            
            # -------- landmarks START --------
            positions_w = []
    
            if '.json' in dataTypeLandmark:
                with open(curPathLandmark) as f:
                    loaded = json.load(f) 
                controlPoints = loaded['markups'][0]['controlPoints']
                for points in controlPoints:
                    curPoints = points['position']
                    positions_w.append(curPoints)   
            elif '.csv' in dataTypeLandmark:
                landmarkCsv = pd.read_csv(curPathLandmark)
            elif '.fcsv' in dataTypeLandmark:
                df = pd.read_csv(curPathLandmark, header=None,  skiprows=[0,1,2])
                for index, row in df.iterrows():
                    positions_w.append([row[1], row[2], row[3]])
            df = None
            
            positions_ijk = tranformPoints_w_ijk( np.concatenate( 
                (positions_w, np.ones((len(positions_w),1))), axis=1),
                niiFile.affine)
            positions_ijk = positions_ijk[:,:3]
             
            if np.any(np.array(positions_ijk) < 0):
                for k in range(len(positions_ijk)):
                    for j in range(len(positions_ijk[k])):
                        if positions_ijk[k][j] < 0:
                            positions_ijk[k][j] = 0

            allPositions_w.append(positions_w)
            allPositions_ijk.append(positions_ijk)

            z_remove = None
            if idxRemoveSections is not None:
                idxRemoveSections_unique = np.unique(np.asarray(idxRemoveSections))
                positionsRemove_ijk = [positions_ijk[i] for i in idxRemoveSections_unique]
                z_remove = [int(x[2]) for x in positionsRemove_ijk]
                
            idxSections_unique = np.unique(np.asarray(idxSections))
            #only consider the sections
            positions_w = [positions_w[i] for i in idxSections_unique]
            positions_ijk = [positions_ijk[i] for i in idxSections_unique]

            if plotLandmarksPng:
                Path(curPngPath).mkdir(parents=True, exist_ok=True)
                curXCoords = np.zeros(len(positions_ijk))
                curYCoords = np.zeros(len(positions_ijk))
                curZCoords = np.zeros(len(positions_ijk))
                for j in range(len(positions_ijk)):
                    curXCoords[j] = int(positions_ijk[j][0])
                    curYCoords[j] = int(positions_ijk[j][1])
                    curZCoords[j] = int(positions_ijk[j][2])

                fig, ax = plt.subplots(1,1)
                ax.imshow(im[:,:,int(curZCoords.mean())], cmap='gray')#, extent=[0, 1, 0, 1] ) 
                ax.scatter( curYCoords, curXCoords,s=10,c='red',marker='*') 
                fig.savefig(curPngPath+curFilename+'_stage0_0_ax.png', dpi=300)
                plt.close(fig) 
                fig, ax = plt.subplots(1,1)
                ax.imshow(im[:,int(curYCoords.mean()),:], cmap='gray')#, extent=[0, 1, 0, 1] ) 
                ax.scatter( curZCoords, curXCoords,s=10,c='red',marker='*') 
                fig.savefig(curPngPath+curFilename+'_stage0_0_cor.png', dpi=300)
                plt.close(fig) 
                fig, ax = plt.subplots(1,1)
                ax.imshow(im[int(curXCoords.mean()),:,:], cmap='gray')#, extent=[0, 1, 0, 1] ) 
                ax.scatter( curZCoords, curYCoords,s=10,c='red',marker='*') 
                fig.savefig(curPngPath+curFilename+'_stage0_0_sag.png', dpi=300)
                plt.close(fig) 
            # -------- landmarks END --------
            
            #Brauche midPonts und normvectors
            #get mid points by clasest point to center line
            idx_closestCenterPos = np.zeros(len(positions_ijk),dtype=int)
            for pos_idx, pos in enumerate(positions_ijk):
                lowest_dist = np.inf
                for center_idx, pos_center in enumerate(centerLineCords_ijk):
                    dist = np.linalg.norm(pos-pos_center)
                    if dist < lowest_dist:
                        lowest_dist = dist
                        idx_closestCenterPos[pos_idx] = int(center_idx)

            midPoints = [ centerLineCords_ijk[idx] for idx in idx_closestCenterPos ]
            normVectors = get_normVectors_by_centerLineIdx( centerLineCords_ijk, idx_closestCenterPos)

            #if section is multiple times in list, then create new positions by defined disatnce
            alreadyInsertedAt = []
            if len(idxSections_unique) < len(idxSections):
                for nameSection in namesSections:
                    if '(mm)' in nameSection:
                        if '-' in nameSection: sign = '-'
                        elif '+' in nameSection: sign = '+'
                        else: 
                            print(f'no -/+ found in {nameSection}. Ignoring this custom landmark...')
                            continue     
                        landmark = nameSection.split(sign)[0]
                        dist = float(nameSection.split('(mm)')[0].split(sign)[-1])
                        #get landmark idx
                        idx_landmark = namesLandmarks.index(landmark)
                        idx_correspondingMidPoint = list(idxSections_unique).index(idx_landmark)
                        idx_centerline = idx_closestCenterPos[idx_correspondingMidPoint]
                        cur_centerLineCoord_w = np.array(centerLineCords_w[idx_centerline])
                        total_dist = 0
                        while total_dist<dist:
                            if sign=='-': 
                                if idx_centerline <= 0: break
                                idx_centerline -= 1
                            else: 
                                if idx_centerline >= len(centerLineCords_w)-1: break
                                idx_centerline += 1
                            last_centerLineCoord_w = cur_centerLineCoord_w
                            cur_centerLineCoord_w = np.array(centerLineCords_w[idx_centerline])
                            total_dist += np.linalg.norm(cur_centerLineCoord_w-last_centerLineCoord_w)  
                            
                        if sign=='-': 
                            insert_idx_orig = idx_correspondingMidPoint
                        else: 
                            insert_idx_orig = idx_correspondingMidPoint + 1
                        insert_idx = insert_idx_orig
                        for already_idx in alreadyInsertedAt:
                            if insert_idx_orig > already_idx:
                                insert_idx += 1
                        alreadyInsertedAt.append(insert_idx_orig)
                        
                        curNormVector = get_normVectors_by_centerLineIdx( centerLineCords_ijk, [idx_centerline])[0]
                        curMidPoint = centerLineCords_ijk[idx_centerline]
                        if idx_centerline == 0 and sign=='-': 
                            #often there is a snipped at the bottom with no centerline point
                            #so go 10mm bit further in neg dir
                            curMidPoint -= 10*curNormVector
                            
                        midPoints.insert(insert_idx, curMidPoint)
                        normVectors.insert(insert_idx, curNormVector)
      
            midPoints = np.array(midPoints)
            normVectors = np.array(normVectors)
            
            new_segm = np.copy(segm).astype(np.uint8)
            
            old_vals_wall = []
            new_vals_wall = []
            fatRegion_mask = np.zeros(new_segm.shape, dtype=bool)
            curFatRegion = np.copy(new_segm)

            kernel = ball_kernel(int(numVoxelOneStep))
            for cur_idxFatRegion in IndiciesStepsFromAorticWall:

                curFatRegion = tensorDilation3D( curFatRegion, kernel)   
                curFatRegion[new_segm>0] = False
                fatRegion_mask += curFatRegion
                #i need to code the values to higher numbers so there is no overlap 
                # of used numbers for the sectioning and if these values are untouched they
                # are automatically the right section
                old_val = cur_idxFatRegion+1
                old_vals_wall.append(old_val)
                new_val = len(midPoints)*(old_val-1)+1
                new_vals_wall.append(new_val)
                new_segm[curFatRegion] = new_val
            
            #if we have cut above, we dont want to have fatRegion within aorta
            new_segm[ np.logical_and(new_segm>len(midPoints), segm) ] = 1
                
            new_segm_mask = new_segm > 0 
            indicies_segm = np.zeros( (np.sum(new_segm_mask), 3))
            indicies_segm[:,0], indicies_segm[:,1], indicies_segm[:,2] = np.where( new_segm_mask)
            indicies_segm = indicies_segm.astype(int)
               
            #Add the rest of the aorta that wont be processed but cut
            new_segm[ np.logical_and(new_segm==0, segm) ] = 1
            new_segm = partitionAtPlanes( new_segm,
                                          indicies_segm,
                                          nb.typed.List(midPoints),
                                          nb.typed.List(normVectors),
                                          nb.typed.List(old_vals_wall),
                                          nb.typed.List(new_vals_wall) ) 

            if z_remove is not None:
                new_segm[:,:,z_remove[0]:z_remove[1]] = 0

            nib.save(nib.Nifti1Image(new_segm, niiFile.affine), curPathSegm_out.replace(dataTypeOut_labels,'_withFat'+dataTypeOut_labels))
            
            new_segm_withOutFat = np.copy(new_segm)
            new_segm_withOutFat[new_segm>len(midPoints)] = 0
            nib.save(nib.Nifti1Image(new_segm_withOutFat, niiFile.affine), curPathSegm_out)
            new_segm_withOutFat = None
            
            # # ---------------- GET FAT ----------------
            fat_segm = np.copy(new_segm)
            fat_mask = np.logical_and(im >= fatWindow[0], im <= fatWindow[1])
            if imCloseFatMask: fat_mask = tensorClose3D(fat_mask, np.ones(kernelSize_imCloseFatMask)).astype(bool)
            fat_segm[ np.logical_or(~fat_mask,~fatRegion_mask) ] = 0
            nib.save(nib.Nifti1Image(fat_segm, niiFile.affine), curPathSegm_out.replace(dataTypeOut_labels,'_fat'+dataTypeOut_labels))
        
        else:
            new_segm = nib.load(curPathSegm_out).get_fdata().astype(np.uint8)  
            fat_segm = nib.load(curPathSegm_out.replace(dataTypeOut_labels,'_fat'+dataTypeOut_labels)).get_fdata().astype(np.uint8)  
        
        # # ---------------- GET KALK ----------------
        if not os.path.isfile(curPathSegm_out.replace(dataTypeOut_labels,'_kalk'+dataTypeOut_labels)) or overWriteStages:
            curKalkThresh = kalkThresh
            kalk_segm = np.copy(new_segm)
            kalk_mask =  im > curKalkThresh 
            if imCloseKalkMask: kalk_mask = tensorClose3D(kalk_mask, np.ones(kernelSize_imCloseKalkMask)).astype(bool)
            kalk_segm[ np.logical_or(~kalk_mask,~segm) ] = 0
            
            if adaptiveKalkThresh:
                
                segm_withoutWall = np.copy(segm)
                if postProc_dilateAorticWall and aorticWallThickenss is not None:         
                    kernel = ball_kernel(int(np.round(aorticWallThickenss/curVoxelSpacing[0])))
                    segm_withoutWall = tensorErosion3D(segm, kernel)
                
                segm_sectioned_mask = np.logical_and( segm_withoutWall, new_segm <= len(midPoints) )
                num_segm_sectioned = segm_sectioned_mask.sum()
                vals_kalk = im[ np.logical_and(segm_withoutWall,
                                np.logical_and( kalk_segm>0,
                                kalk_segm<=len(midPoints)) ) ]
                num_kalk = vals_kalk.shape[0]
                curPercOfVolumeInThresh = num_kalk/num_segm_sectioned
                if curPercOfVolumeInThresh > percOfVolumeInThresh/100: #adapt thresh!
                    vals_segm_sectioned = im[segm_sectioned_mask]
                    curKalkThresh = np.percentile(vals_segm_sectioned, 100-percOfVolumeInThresh)
                    kalk_mask =  im > curKalkThresh 
                    #do imclose with small kernel so single voxels due to noise are excluded
                    # kalk_mask = ndimage.binary_closing(kalk_mask, structure=np.ones((3,3,3), dtype=bool) )
                    if imCloseKalkMask: kalk_mask = tensorClose3D(kalk_mask, np.ones(kernelSize_imCloseKalkMask)).astype(bool)
                    kalk_segm = np.copy(new_segm)
                    kalk_segm[ np.logical_or(~kalk_mask,~segm) ] = 0
          
            kalk_segm[kalk_segm>len(midPoints)] = 0 #only kalk in aorta not in fatRegions
            nib.save(nib.Nifti1Image(kalk_segm, niiFile.affine), curPathSegm_out.replace(dataTypeOut_labels,'_kalk'+dataTypeOut_labels))
    
            kalkThreshUsed.append(curKalkThresh)
            with open(curPathIm.replace(dataTypeIn,'_kalk_thresh.txt'), 'w') as text_file:
                print(str(curKalkThresh), file=text_file)
        else:
            kalk_segm = nib.load(curPathSegm_out.replace(dataTypeOut_labels,'_kalk'+dataTypeOut_labels)).get_fdata().astype(np.uint8)
            with open(curPathIm.replace(dataTypeIn,'_kalk_thresh.txt')) as text_file:
                curKalkThresh = float(text_file.read().replace('\n',''))
        
        # # ---------------- GET METRICS ----------------
        
        vals = [ curFilename, curVoxelSpacing[0], curVoxelSpacing[1], curVoxelSpacing[2], numVoxelOneStep,
                im_shape[0], im_shape[1], im_shape[2], curKalkThresh, fatWindow[0], fatWindow[1]]
        keys = ['name','voxSpacing_x','voxSpacing_y','voxSpacing_z', 'numVoxelOneStep',
                'matSize_x','matSize_y','matSize_z', 'thresh_kalk', 'thresh_fat_0', 'thresh_fat_1']
    
        all_valsForFAIPlot_mean = []
        all_valsForFAIPlot_std = []
        for name_idx, name in enumerate(namesSections):
            #### METRICS FOR SEGMENT
            curLabel = name_idx+1
            curImageVals = im[new_segm==curLabel]
            
            curVol = len(curImageVals)*curFactorNumToVol
            if curVol > 0:
                curMean = np.nanmean(curImageVals)
                curStd = np.nanstd(curImageVals)
                curMedian = np.nanmedian(curImageVals)
            else:
                curMean = np.nan
                curStd = np.nan
                curMedian = np.nan
                
            keys.append(name+'_volume')
            vals.append( curVol )
            keys.append(name+'_meanHU')
            vals.append( curMean )
            keys.append(name+'_stdHU')
            vals.append(  curStd )
            keys.append(name+'_medianHU')
            vals.append( curMedian )
            
            #### METRICS FOR KALK
            curImageVals = im[kalk_segm==curLabel]
        
            curVol = len(curImageVals)*curFactorNumToVol
            if curVol > 0:
                curMean = np.nanmean(curImageVals)
                curStd = np.nanstd(curImageVals)
                curMedian = np.nanmedian(curImageVals)
            else:
                curMean = np.nan
                curStd = np.nan
                curMedian = np.nan
        
            keys.append(name+'_kalk_volume')
            vals.append( curVol )
            keys.append(name+'_kalk_meanHU')
            vals.append( curMean )
            keys.append(name+'_kalk_stdHU')
            vals.append( curStd )
            keys.append(name+'_kalk_medianHU')
            vals.append( curMedian )

            #### METRICS FOR FAT
            valsForFAIPlot_mean = [] 
            valsForFAIPlot_std = []
            
            for curRange_idx, curRange in enumerate(IndiciesStepsFromAorticWall):

                curLabel = (len(namesSections)*curRange_idx)+len(namesSections)+1+name_idx
                curImageVals = im[fat_segm==curLabel]
                
                curVol = len(curImageVals)*curFactorNumToVol
                if curVol > 0:
                    curMean = np.nanmean(curImageVals)
                    curStd = np.nanstd(curImageVals)
                    curMedian = np.nanmedian(curImageVals)
                else:
                    curMean = np.nan
                    curStd = np.nan
                    curMedian = np.nan
              
                keys.append(name+'_fat_'+str(curRange)+'_volume')
                vals.append( curVol )
                keys.append(name+'_fat_'+str(curRange)+'_meanHU')
                vals.append( curMean )
                keys.append(name+'_fat_'+str(curRange)+'_stdHU')
                vals.append( curStd )
                keys.append(name+'_fat_'+str(curRange)+'_medianHU')
                vals.append( curMedian )
                
                valsForFAIPlot_mean.append(curMean)
                valsForFAIPlot_std.append(curStd)
            
            if IndiciesStepsFromAorticWall[-1] < numSteps_total:
                for curRange in range(int(IndiciesStepsFromAorticWall[-1])+1, numSteps_total+1):
                    curRange = int(curRange)
                    keys.append(name+'_fat_'+str(curRange)+'_volume')
                    vals.append( np.nan )
                    keys.append(name+'_fat_'+str(curRange)+'_meanHU')
                    vals.append( np.nan )
                    keys.append(name+'_fat_'+str(curRange)+'_stdHU')
                    vals.append( np.nan )
                    keys.append(name+'_fat_'+str(curRange)+'_medianHU')
                    vals.append( np.nan )

            all_valsForFAIPlot_mean.append( deepcopy(valsForFAIPlot_mean))
            all_valsForFAIPlot_std.append( deepcopy(valsForFAIPlot_std))
    
        if group_a_appendix in curFilename: 
            if dict_out_a is None: dict_out_a = createDictFromKeysAndVals(keys,vals)
            else: dict_out_a = appendToDictFromKeysAndVals(dict_out_a,keys,vals)
        else:
            if dict_out_b is None: dict_out_b = createDictFromKeysAndVals(keys,vals)
            else: dict_out_b = appendToDictFromKeysAndVals(dict_out_b,keys,vals)
            
        fig, ax = plt.subplots(1,1)
        distancesFromAorticWall = IndiciesStepsFromAorticWall*curVoxelSpacing[0]*numVoxelOneStep
        all_valsForFAIPlot_mean = [ np.asarray(x) for x in all_valsForFAIPlot_mean]
        all_valsForFAIPlot_std = [ np.asarray(x) for x in all_valsForFAIPlot_std]
        
        for name, color, curMean, curStd in \
            zip(namesSections, colorsPlot, all_valsForFAIPlot_mean, all_valsForFAIPlot_std):  
            ax.plot(distancesFromAorticWall, curMean, color, label=name)
            if showErrorBands:
                ax.fill_between(distancesFromAorticWall, curMean - curStd,
                                curMean + curStd,
                                color='dimgray', alpha=alpha_errorBands) 
        
        ax.legend()
        ax.set_xlabel('distance to vessel [mm]')#, fontsize=cfg.fontsize_axis)
        ax.set_ylabel('mean HU')#, fontsize=cfg.fontsize_axis)
        fig.savefig(curPngPath+curFilename+'_FAI_plot.png', dpi=300)
        plt.close(fig)
        
    df_out_a =  pd.DataFrame(dict_out_a)
    df_out_a.to_excel(inputPath+group_a_appendix+'results.xlsx')
    df_out_b =  pd.DataFrame(dict_out_b)
    df_out_b.to_excel(inputPath+group_b_appendix+'results.xlsx')

    voxelSpacings = np.asarray(voxelSpacings)
    matrixSizes = np.asarray(matrixSizes)
    kalkThreshUsed = np.asarray(matrixSizes)

    # Saving the objects:
    with open(stage0_pkl, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([ filenames, pathsSegm, pathsIm, pathsLandmarks, voxelSpacings, matrixSizes, kalkThreshUsed,
                      cutCoordZ, allPositions_ijk, allPositions_ijk_cutZ, allPositions_w, allPositions_w_cutZ], f)
else:
    
    # Getting back the objects from stage 0:
    with open(stage0_pkl, 'rb') as f:  # Python 3: open(..., 'rb')
        [ filenames, pathsSegm, pathsIm, pathsLandmarks, voxelSpacings, matrixSizes, kalkThreshUsed,
        cutCoordZ, allPositions_ijk, allPositions_ijk_cutZ, allPositions_w, allPositions_w_cutZ] = pickle.load(f)

sns.reset_orig()
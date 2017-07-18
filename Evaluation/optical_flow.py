#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 03:51:29 2017

@author: Arpan

Description: Extract the Optical Flow data from action videos

"""
import cv2
import numpy as np
import os
import json
import lmdb
import caffe
import pandas as pd
from matplotlib import pyplot as plt
from joblib import Parallel, delayed


# Input: 
# srcVideoFolder: where the action videos are located (for train/val/test set)
# Output: Create optical flow visualization data, transformed to HSV space
# ToDo: write the feature onto a file and convert to lmdb.
def construct_datasets(srcVideoFolder, lmdb_folder, pathPrefix, \
                       samples_files, category_names):
    
    DIFF_FRAMES = [1]
    print("No of samples_files = "+str(len(samples_files)))      # =no_of_categories
    lmdb_name = os.path.join(lmdb_folder, "val_OF_lmdb")    
    if not os.path.exists(os.path.dirname(lmdb_name)):
        os.makedirs(os.path.dirname(lmdb_name))
    
    # form a pandas dataframe with video_id 
    video_id, pos, labels = [], [], []
    for idx,f in enumerate(samples_files):
        if category_names[idx] in f:
            with open(os.path.join(pathPrefix, f), "r") as fobj:
                pos_samples = json.load(fobj)
            for v_id, pos_list in pos_samples.iteritems():
                pos.extend(pos_list)
                video_id.extend(np.repeat(v_id, len(pos_list)).tolist())
                labels.extend(np.repeat(idx, len(pos_list)).tolist())
    samples_df = pd.DataFrame({'video_id': video_id,
                               'position': pos,
                               'label': labels})
    print "No of samples for all the categories = {} " .format(samples_df.shape[0]) 
    
    # Shuffle the dataframe in-place
    samples_df = samples_df.sample(frac=1).reset_index(drop=True)
    # write dataframe to disk (csv)
    samples_df.to_csv(os.path.join(lmdb_folder, "samples_val.csv"), index=False)
    
    # Create lmdb
    (H, W, C) = (120, 160, 3)
    N = samples_df.shape[0]     # no of rows (=no of visualizations = 5k)
    # twice the size of total number of OF visualizations
    map_size = int(N*H*W*C*3)     # approx 429 GB
    #map_size = int(N*720*1280*C*2)     # approx 429 GB
    
    env = lmdb.open(lmdb_name, map_size=map_size)
    
    i = 0   # LMDB index variable
    # iterate over the rows of the pandas dataframe
    end_samples = samples_df.shape[0]
    r = (end_samples - i)/200
    print "r = %d " %r
    ###########################################################################
    nCat = 4*len(category_names)          # = 200
    nCat_samples = (end_samples - i)/nCat    # = N = 1000
    lmdb_id = 0
                   
    # Parallelizing the lmdb creation process
    for i in range(nCat_samples):
        
        result = Parallel(n_jobs=4)(delayed(get_optical_flow_vid) \
                          (os.path.join(srcVideoFolder, 'v_'+samples_df['video_id'][i*nCat+j]+'.mp4'), \
                           samples_df['position'][i*nCat+j], \
                            DIFF_FRAMES, H, W) \
                          for j in range(nCat))
        
        with env.begin(write = True) as txn:
            for l in range(len(result)):
                row_no = (i*nCat)+l
                pos = samples_df['position'][row_no]
                video_id = samples_df['video_id'][row_no]
                lab = samples_df['label'][row_no]
                print "idx : "+str(row_no)+" :: 'position' : "+str(pos)
                
                for img in result[l]:
                    img = np.rollaxis(img, 2)   # C, H, W
                    datum = caffe.proto.caffe_pb2.Datum()
                    datum.channels = img.shape[0]
                    datum.height = img.shape[1]
                    datum.width = img.shape[2]
                    datum.data = img.tobytes()
                    datum.label = lab
                    str_id = '{:08}'.format(lmdb_id)
                    # The encode is only essential in Python 3
                    txn.put(str_id.encode('ascii'), datum.SerializeToString())
                    lmdb_id += 1
        print "Write No : %d" %(i+1)
    ###########################################################################    
#    for commit_no in range(r):
#        with env.begin(write=True) as txn:    
#            for idx in range(200):  # samples_df.iterrows():
#                row_no = (200*commit_no)+idx
#                assert i==row_no
#                pos = samples_df['position'][row_no]
#                video_id = samples_df['video_id'][row_no]
#                lab = samples_df['label'][row_no]
#                print "idx : "+str(row_no)+" :: 'position' : "+str(pos)
#                imgs = []
#                vpath = os.path.join(srcVideoFolder, 'v_'+video_id+'.mp4')
#                imgs.extend(get_optical_flow_vid(vpath, pos, DIFF_FRAMES, H, W))
#                # returned frames are HxWxC (120x160x3) in a list
#            
#                for img in imgs:
#                    # rollaxis if needed
#                    img = np.rollaxis(img, 2)   # C, H, W
#                    datum = caffe.proto.caffe_pb2.Datum()
#                    datum.channels = img.shape[0]
#                    datum.height = img.shape[1]
#                    datum.width = img.shape[2]
#                    datum.data = img.tobytes()
#                    datum.label = lab
#                    str_id = '{:08}'.format(i)
#                    # The encode is only essential in Python 3
#                    txn.put(str_id.encode('ascii'), datum.SerializeToString())
#                    i = i+1
        
    print "LMDB Created Successfully !!"
    return 
    
# from a srcVideo, get the optical flow data of ith and (i+x) frame 
# where x belongs to diff_frames
def get_optical_flow_vid(srcVideo, position, diff_frames, height, width):
    res_flow_img = []
    cap = cv2.VideoCapture(srcVideo)
    #fgbg = cv2.createBackgroundSubtractorMOG2()     #bg subtractor
    if not cap.isOpened():
        raise IOError("Capture object cannot be opened for "+srcVideo)
    ####################################################
    # for resizing the optical flow visualization
    resize_flag = True
    (h, w) = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),\
                  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    if h==height and w==width:
        resize_flag = False
    
    #print "No of frames = {}", format(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for diff in diff_frames:
        #print "For diff = %d" %diff
        cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        ret, frame = cap.read()
        # Sometimes the last few frames of a video are not read, then read the
        # last readable frame by moving backwards one frame at a time
        while not ret:
            #print "Frame not read ! Moving backwards in capture object !"
            #raise IOError("Frame not read :: "+srcVideo+" :: Position: "+str(position))
            position -= 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, position)
            ret, frame = cap.read()
    
#        curr_frame = frame.copy()
#        cap.set(cv2.CAP_PROP_POS_FRAMES, position+diff)
#        ret, next_frame = cap.read()
#        # If next frame is unavailable, then make cf as nf and read previous frame in cf
#        if not ret:
#            #print "Cannot read next frame... Reading previous frame instead."
#            cap.set(cv2.CAP_PROP_POS_FRAMES, position-diff)
#            next_frame = curr_frame.copy()
#            ret, curr_frame = cap.read()
#            if not ret:
#                raise IOError("Cannot read previous frame also.")
#    
#        curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
#        next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
#        # Compute the optical flow        
#        flow = cv2.calcOpticalFlowFarneback(curr_frame, next_frame, None, 0.5, 1, 12, 3, 5, 1.2, 0)
#        #vis_vectors = draw_flow(curr_frame, flow, 8)
#        vis_bgr = draw_flow_bgr(flow, frame)
    
        if resize_flag:
            # scaling image. Mostly it scales down to 120x160 (hxw) INTER_LINEAR default
            frame = cv2.resize(frame, (width, height) )
        
        res_flow_img.append(frame)
        #cv2.imshow("Curr Frame", curr_frame)
        #cv2.imshow("Next Frame", next_frame)
        #cv2.imshow("Flow Vecs", vis_vectors)
        #cv2.imshow("Flow BGR", vis_bgr)
        #waitTillEscPressed()
        
    #res_mean = []
    #res_mean.append(np.average(res_flow_img, axis=0).astype(np.uint8))
    
    cap.release()
    #cv2.destroyAllWindows()
    return res_flow_img


# draw the OF field on image, with grids, decrease step for finer grid
def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def draw_flow_bgr(flow, sample_frame):
    hsv = np.zeros_like(sample_frame)
    #print "hsv_shape : "+str(hsv.shape)
    hsv[...,1] = 255
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

    
def waitTillEscPressed():
    while(True):
        if cv2.waitKey(10)==27:
            print("Esc Pressed")
            return

    
if __name__=="__main__":
    # the dataset folder contains 6 folders boxing, running etc containing videos for each
    # It also contains 00sequences.txt where meta info is given
    dataset = "/home/hadoop/VisionWorkspace/ActivityNet/ActivityNet-master/Crawler/videos"

    srcVideo = os.path.join(dataset, 'v_2GEZgHcA7zU.mp4')
    
#    img = get_optical_flow_vid(srcVideo, 2984, [1,2,3], 120, 160)
#    for i,im in enumerate(img):
#        print "Flow image no : %d" %(i+1)
#        cv2.imshow("Frame", im)
#        waitTillEscPressed()
    lmdb_folder = "/home/hadoop/VisionWorkspace/ActivityNet"
    p = "/home/hadoop/VisionWorkspace/ActivityNet/ActivityNet-master/Evaluation/samples_test_5000"
    sampls = os.listdir(p)
    construct_datasets(dataset, lmdb_folder, p, sampls, ['Applying sunscreen'])
    #cv2.destroyAllWindows()
    
    ###########################################################
    # Training the caffe model    
    #proc = subprocess.Popen(["/home/hadoop/caffe/build/tools/caffe","train","--solver=optical_flow_lenet_solver.prototxt"],stderr=subprocess.PIPE)
    #res = proc.communicate()[1]

    #caffe.set_mode_gpu()
    #solver = caffe.get_solver("config.prototxt")
    #solver.solve()
    
    #print res
    ###########################################################
    # Applying the model
    
    #net = caffe.Net("demoDeploy.prototxt", "./opt_flow_quick_iter_20000.caffemodel", caffe.TEST)
    #print(get_data_for_id_from_lmdb("/home/lnmiit/caffe/examples/optical_flow/val_opt_flow_lmdb/", "00000209"))
    #l, f = get_data_for_id_from_lmdb("/home/lnmiit/caffe/examples/optical_flow/val_opt_flow_lmdb/", "00000209")
    
    ###########################################################
    ## Check Background Subtraction on sample videos (Visualize)
#    srcVideo = "/home/hadoop/VisionWorkspace/KTH_OpticalFlow/dataset/kth_actions_test/person03_walking_d1_uncomp.avi"
#    cap = cv2.VideoCapture(srcVideo)
#    fgbg = cv2.createBackgroundSubtractorMOG2()
#    while(cap.isOpened()):
#        ret, frame = cap.read()
#        fgmask = fgbg.apply(frame)
#        cv2.imshow('frame',fgmask)
#        print np.sum(fgmask)
#        waitTillEscPressed()
#        #k = cv2.waitKey(30) & 0xff
#        #if k == 27:
#        #    break
#    cap.release()
#    cv2.destroyAllWindows()

    
    
    
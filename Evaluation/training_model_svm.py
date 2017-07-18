#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 20:53:14 2017

@author: Arpan

Description: Training Models

"""
import json
import os
import utils
import numpy as np
import cv2

def get_hog(srcVideo, start, end):
    return

# To train a single SVM which identifies one class, get  +ve samples frames
# and get same amount of -ve sample frames
    
def get_meta_info(video_path, existing_vids):
    """Add meta information of existing training videos to a dictionary and 
    write the dictionary to a file.
    
    Input: existing_vids: Videos Ids of the mp4 files. 
        Note that only the training video Ids should be sent here
    Return: dictionary containing the video_ids as keys and corresponding
        meta-info
    """
    meta_dict = {}
    # loop over the VideoIDs and get the meta information for each file
    print "Getting video meta-information..."
    for v in existing_vids:
        filePath = os.path.join(video_path, "v_"+v+".mp4")
        cap = cv2.VideoCapture(filePath)
        if not cap.isOpened():
            raise IOError("Capture object not opened ! Abort !")
            break
        fps = cap.get(cv2.CAP_PROP_FPS)
        # dimensions = (Ht, Wd)
        dimensions = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), \
                      int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        no_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        meta_dict[v] = {"fps":fps, "dimensions":dimensions, \
                 "total_frames": no_of_frames}
        cap.release()

    return meta_dict
    
def get_training_segments(database, video_ids_for_cat, category):
    """
    Get training segments from the videos and form a dictionary
    Note: It applies for +ve examples as of now.
    """
    segments_dict = {}
    start, stop = 0, 0
    for v in video_ids_for_cat:
        # list of annotations on video
        annotations = database[v]["annotations"]
        for ann in annotations:
            if ann["label"] == category:
                start, stop = ann["segment"]
                if v in segments_dict:
                    segments_dict[v].append({"start": start, "stop": stop})
                else:
                    segments_dict[v] = [{"start": start, "stop": stop}]
        # for a dictionary of segments, with key as video id and 
        # values as the list of start and stop times of +ve examples
    return segments_dict
    

def get_sample_frames(seg, meta_info, N):
    """ Get N sample frames from the defined video segments of the given 
    video_ids. 
    Input:
    seg: (Dictionary) Training segments for positive example videos for single
         category.
        {"FKQIdqjY9nI": [{'start': 12.73, 'stop': 22.23} ... ]}
    meta_info: dict for meta_info of all existing training videos
            {"FKQIdqjY9nI": {'total_frames': 1056, 
            'dimensions': (720, 1280), 'fps': 30.0} ...}
    N : Total number of samples to be extracted
    Output: 
        pos_samples: {"FKQIdqjY9nI": [ 234, 543], ...}
    """
    # Get total number of frames in all the segments across all videos
    # Get N samples from total number of frames
    # Map the generated integers backwards to the frame numbers of video segments
    # Get the video_id, frame number that needs to be sampled
    total_frames = 0
    # Iterate over all the segments of the videos containing actions
    video_ids = sorted(seg.keys())
    for v_id in video_ids:
        for segment in seg[v_id]:
            frames_in_seg = int((segment["stop"] - segment["start"])*meta_info[v_id]["fps"])
            total_frames += frames_in_seg
        
    print "Total frames in all segments = %d " % total_frames
    # Randomly (uniform) sample N values from 0 to total_frames-1
    # Backwards mapping
    import random
    random.seed(231)
    samp = sorted(random.sample(range(1, total_frames), N), reverse=True)
    #print "Samples list !! "
    #print samp
    pos_samples = {}
    frame_ptr_lower = 0
    for v_id in video_ids:
        for segment in seg[v_id]:
            frames_in_seg = int((segment["stop"]-segment["start"])*meta_info[v_id]["fps"])
            #print "v_id %s || Frames in seg : %d || lower : %d" %(v_id, frames_in_seg, frame_ptr_lower)
            while len(samp)!=0 and (frame_ptr_lower<=samp[-1] \
                     and samp[-1]<=(frame_ptr_lower+frames_in_seg)):
                samp_no = samp.pop()
                # Pop until the popped item is not in range
                # Get no of frames in video segment using video's FPS 
                # calculate position (Frame number) in the video and write to dict
                pos = int(segment["start"]*meta_info[v_id]["fps"])+(samp_no-frame_ptr_lower)
                #print "lower : %d || samp_no : %d || pos : %d " %(frame_ptr_lower, samp_no, pos)
                if v_id in pos_samples:
                    pos_samples[v_id].append(pos)
                else:
                    pos_samples[v_id] = [pos]
            frame_ptr_lower += frames_in_seg
            
    #print "Samples information written to dictionary with size: %d" %len(pos_samples)
    return pos_samples
               
def display_sample_frames(samples_dict, srcFolder):
    """
    Display the frames from the samples dictionaries of the categories
    Input:
        samples_dict: {"FKQIdqjY9nI": [ 234, 543], ...}
        srcFolder : path containing the videos
    """
    # Loop over the videos and display the frames
    
    for v_id in samples_dict:
        cap = cv2.VideoCapture(os.path.join(srcFolder, "v_"+v_id+".mp4"))
        if not cap.isOpened():
            raise IOError("Capture object not opened !")
        pos_lst = samples_dict[v_id]
        for pos in pos_lst:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            cv2.imshow("Frame", frame)
            waitTillEscPressed()
        cap.release()
    cv2.destroyAllWindows()
    return
    

def get_negative_frames(seg, meta_info, N, category):
    """
        Get N samples from videos that do not belong to the segments mentioned in
        seg and are not of 'category'. 
    """
    return 

def train_svm(srcVideo, annotations, incr_rate, category_names):
    cap = cv2.VideoCapture(srcVideo)
    
    if not cap.isOpened():
        raise IOError("Video cannot be opened !")
    
    dimensions = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print "Dimensions : %s " % str(dimensions)
    print "Frame Rate : %f " % fps
    # Loop over the annotation dictionaries
    for ann in annotations:
        start_time, stop_time = ann['segment']
        start = int(start_time*fps)
        stop = int(stop_time*fps)
        label = ann['label']
        print "Action Label : %s" %label
        while cap.isOpened() and start<stop:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if ret:
                #frame = cv2.resize(frame )
                cv2.imshow("Action", frame)
                print "Frame no : %d " % start
                waitTillEscPressed()
                start += incr_rate
                continue
            else:
                break
    cap.release()
    cv2.destroyAllWindows()
    return

def waitTillEscPressed():
    while(True):
        # For moving forward
        if cv2.waitKey(10)==27:
            print("Esc Pressed. Move Forward without labeling.")
            return 1

# for testing the functions
if __name__=='__main__':
    VIDEOPATH = '/home/hadoop/VisionWorkspace/ActivityNet/ActivityNet-master/Crawler/videos'
    srcVideo = ''
    #test_frame_rate(VIDEOPATH)
    samps = {}
    #samps['x_luDzL03vw'] = []
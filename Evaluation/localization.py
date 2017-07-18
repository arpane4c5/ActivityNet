#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 14:13:27 2017

@author: Arpan
Description : For Localization Task
"""
import json
import os
import utils
import numpy as np
import h5py
import pandas as pd
import collections
import cPickle
import caffe
from joblib import Parallel, delayed

VIDEOPATH = '/home/hadoop/VisionWorkspace/ActivityNet/ActivityNet-master/Crawler/videos'
ANNOTATION_FILE = '/home/hadoop/VisionWorkspace/ActivityNet/ActivityNet-master/Evaluation/data/activity_net.v1-3.min.json'
PROPOSALS_FILENAME = '/home/hadoop/VisionWorkspace/ActivityNet/Downloads/Temporal Activity Proposals/activitynet_v1-3_proposals.hdf5'
SHUFFLE = '/home/hadoop/VisionWorkspace/ActivityNet/Downloads/ImageNet Shuffle Features/ImageNetShuffle2016_features.h5'
MBH = "/home/hadoop/VisionWorkspace/ActivityNet/Downloads/MBH Features/MBH_Videos_features.h5"
C3D = "/home/hadoop/VisionWorkspace/ActivityNet/Downloads/C3D Features/sub_activitynet_v1-3.c3d.hdf5"
C3D_PCA = "/home/hadoop/VisionWorkspace/ActivityNet/Downloads/C3D Features/PCA_activitynet_v1-3.hdf5"
SHUFFLE_IDS = '/home/hadoop/VisionWorkspace/ActivityNet/Downloads/ImageNet Shuffle Features/ImageNetShuffle2016_quids.txt'
LMDB_FOLDER = "/home/hadoop/VisionWorkspace/ActivityNet/new_lmdb"
MODEL = "/home/hadoop/VisionWorkspace/ActivityNet/ActivityNet-master/caffe_models/deploy_c3d_fc_net.prototxt"
PRETRAINED = "/home/hadoop/VisionWorkspace/ActivityNet/ActivityNet-master/caffe_models/snapshots/c3d_4k_1k/c3d_fc_net_snap_iter_400000.caffemodel"
MEANFILE = "/home/hadoop/VisionWorkspace/ActivityNet/ActivityNet-master/caffe_models/mean_c3d_4k.binaryproto"
SUBSET = 'validation'

#def get_temporalProps(net, test_vids, meta_info, category_names, n):
#    fc3d = h5py.File(C3D, 'r')
#    fpca = h5py.File(C3D_PCA, 'r')
#    train_mean = get_training_mean(MEANFILE)
#    pred = {}
#    c3d_lev2 = pd.DataFrame(np.zeros((len(test_vids), len(category_names))), \
#                            index=test_vids, columns=category_names)
#    for i, vid in enumerate(test_vids):
#        print "{} --> For video : {}" .format(i, vid)
#        (rows, cols) = fc3d['v_'+vid]['c3d_features'].shape
#        vid_data = fc3d['v_'+vid]['c3d_features'][:]
#        # get predictions for each row of c3d feature of vid
#        vid_probs = pd.DataFrame(np.zeros((rows, len(category_names))), \
#                                 columns=category_names)
#        predicted_labels = []
#        for row in range(rows):
#            #print "Dims of vid_data[row,:] = {}" .format(vid_data[row,:].shape)
#            #print "Values = {}" .format(vid_data[row,:])
#            f = vid_data[row,:].reshape(cols, 1, 1)
#            #print "Values = {}" .format(f)
#            f = f - train_mean
#            #pr = net.forward()
#            out = net.forward_all(data = np.asarray([f]))
#            predicted_labels.append(out['prob'][0].argmax(axis=0))
#            #print "Predicted Label : {} :: Name : {}" .format(predicted_label, category_names[predicted_label])
#            #print "Rows :: "
#            vid_probs.iloc[row,:] = out['prob'][0]
#            #print vid_probs.iloc[row,:]
#        # returns a list of dict like [{'score': score, 'label':labels[idx]}...]
#
#        pred[vid], vprobs = get_vidProposal(vid, vid_probs, predicted_labels,\
#                            meta_info, category_names, n)
#        c3d_lev2.loc[vid,:] = vprobs
#        #break
#    fc3d.close()
#    fpca.close()
#    return pred, c3d_lev2

def get_vidLocalization(vid, vid_probs, vid_preds, meta_info, category_names, n):
    """Get a matrix of probabilities over the classes for the c3d features of 
        a video. Generate the top 3 predictions from the prob matrix
        vid_probs: matrix of probs t x C. where t is the no of c3d features per vid
        and C is the no of classes
        vid_preds: list of predictions for that video (vid). len(vid_preds) = t
        and predicted class at position i is category_names[vid_preds[i]]
        
    """
    anno_list = []
    #n = 1       # Taking top n categories
    vprobs_sum = vid_probs.sum(axis=0)
    top_n = vprobs_sum.sort_values(ascending = False)[:n]
    topn_labels = top_n.index.tolist()
    topn_idx = [category_names.index(l) for l in topn_labels]
    # Idea 2 : Detect temporal continuity of category predicted. Longer the better
    #print "Predictions list : {}" .format(vid_preds)
    # find the max number of occurences for any class
    #counter = collections.Counter(vid_preds)
    #top_n = counter.most_common(3)      # get a list of tuples 
    #fps = 27.673616877683916     # mean fps of all vids in training set
    fps = 29.970029970029969     # median 29.970029970029969 3018 times
    if vid in meta_info.keys():
        fps = meta_info[vid]['fps']
    for idx in range(n):
        # get list of tuples (beg_pos, end_pos)
        segments = get_segments_for_cat(vid_preds, topn_idx[idx]) 
        ##### get time in sec from video info
        if len(segments)>0:
            for (beg,end) in segments:
                begtime = (beg+1)*8./fps
                endtime = (end+1)*8./fps
                # taking score as the temporal extent of the activity of interest
                anno_list.append({'label': topn_labels[idx] ,'score': end-beg,\
                                  'segment': [begtime, endtime]})
    
    # Find the top predicted label
    return anno_list, vprobs_sum

#def get_segments_for_cat(pred_lst, cat_id, nth_val):
#    """Retrieve segments corresponding to category number 'cat_id' from the list of 
#    category predictions 'pred_lst'. Return a list of segment tuples
#    """
#    int_seg_dist = 20    # 2 for 8*i frames
#    seg_len_th = 3 - nth_val
#    segments = []
#    beg , end = -1, -1
#    seg_flag = False
#    for i,pr in enumerate(pred_lst):
#        if pr==cat_id and not seg_flag:
#            beg = i
#            seg_flag = True
#        elif pr!=cat_id and seg_flag:
#            end = i
#            segments.append((beg, end))
#            seg_flag = False
#            beg, end = -1, -1
#    if seg_flag:
#        segments.append((beg, i+1))
#    
#    seg_flag = True
#    merged_segments, new_segments = [], []
#    if len(segments)==0:
#        return []
#    (bPrev, ePrev) = segments[0]
#    # Merge 'close' segments based on int_seg_dist
#    for i,(bCurr,eCurr) in enumerate(segments):
#        if i==0:
#            continue
#        if (bCurr-ePrev)>int_seg_dist :
#            merged_segments.append((bPrev, ePrev))
#            bPrev = bCurr
#        ePrev = eCurr
#    merged_segments.append((bPrev, ePrev))
#    # Create a dict of segment lengths for each tuple in segments
#    seg_lens = {}
#    for idx,seg in enumerate(merged_segments):
#        seg_lens[idx] = seg[1] - seg[0]
#    # get segment idxs in decreasing order of lens (sort dict values and get keys)
#    decr_seg_lens = sorted(seg_lens, key=seg_lens.get, reverse=True)
#    # For very small length videos
#    if len(pred_lst) < 3 or len(decr_seg_lens) < 3:
#        seg_len_th = 0
#
#    for idx in decr_seg_lens:
#        if seg_lens[idx] < seg_len_th:
#            break
#        new_segments.append(merged_segments[idx])
#        
#    return new_segments

def get_segments_for_cat(pred_lst, cat_id):
    """Retrieve segments corresponding to category number 'cat_id' from the list of 
    category predictions 'pred_lst'. Return a list of segment tuples
    """
    int_seg_dist = 60    # 2 for 8*i frames
    segments = []
    beg , end = -1, -1
    seg_flag = False
    for i,pr in enumerate(pred_lst):
        if pr==cat_id and not seg_flag:
            beg = i
            seg_flag = True
        elif pr!=cat_id and seg_flag:
            end = i
            segments.append((beg, end))
            seg_flag = False
            beg, end = -1, -1
    if seg_flag:
        segments.append((beg, i+1))
    
    seg_flag = True
    new_segments = []
    if len(segments)==0:
        return []
    (bPrev, ePrev) = segments[0]
    for i,(bCurr,eCurr) in enumerate(segments):
        if i==0:
            continue
        if (bCurr-ePrev)<=int_seg_dist :
            ePrev = eCurr
        else:
            new_segments.append((bPrev, ePrev))
            bPrev = bCurr
            ePrev = eCurr
    new_segments.append((bPrev, ePrev))
        
    return new_segments

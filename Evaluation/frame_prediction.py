#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 14:49:28 2017

@author: Arpan
Description: Use c3d trained model for prediction. To be executed after 
training_model_m4.py
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


# Temporal Proposals : Pretrained
#VIDEOPATH = '/home/arpan/DATA_Drive/ActivityNet/videos'
#ANNOTATION_FILE = '/home/arpan/DATA_Drive/ActivityNet/ActivityNet-master/Evaluation/data/activity_net.v1-3.min.json'
#PROPOSALS_FILENAME = '/home/arpan/DATA_Drive/ActivityNet/extra_features/Temporal Activity Proposals/activitynet_v1-3_proposals.hdf5'
#SHUFFLE = '/home/arpan/DATA_Drive/ActivityNet/extra_features/ImageNet Shuffle Features/ImageNetShuffle2016_features.h5'
#MBH = "/home/arpan/VisionWorkspace/ActivityNet/MBH Features/MBH_Videos_features.h5"
#MBH_IDS = "/home/arpan/VisionWorkspace/ActivityNet/MBH Features/MBH_Videos_quids.txt"
#C3D = "/home/arpan/DATA_Drive/ActivityNet/extra_features/C3D/sub_activitynet_v1-3.c3d.hdf5"
#C3D_PCA = "/home/arpan/DATA_Drive/ActivityNet/extra_features/C3D/PCA_activitynet_v1-3.hdf5"
#SHUFFLE_IDS = '/home/arpan/DATA_Drive/ActivityNet/extra_features/ImageNet Shuffle Features/ImageNetShuffle2016_quids.txt'
#MODEL = "/home/arpan/DATA_Drive/ActivityNet/ActivityNet-master/caffe_models/deploy_c3d_fc_net.prototxt"
#PRETRAINED = "/home/arpan/DATA_Drive/ActivityNet/ActivityNet-master/caffe_models/snapshots/c3d_4k_1k/c3d_fc_net_snap_iter_400000.caffemodel"
#MEANFILE = "/home/arpan/DATA_Drive/ActivityNet/ActivityNet-master/caffe_models/mean_c3d_4k.binaryproto"
#SUBSET = 'validation'

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
SUBSET = 'testing'


def get_c3d_feature(fc3d, vid, pos, vfps):
    '''
    Read the feature vector that is near the pos of video
    c3d features are taken for every 8th frame
    '''
    row = int(pos/8)
    while not row <= fc3d[vid]['c3d_features'].shape[0]:
        print "Decrement by 1"
        row -= 1
        assert row <= fc3d[vid]['c3d_features'].shape[0]
    vec = fc3d[vid]['c3d_features'][row,:]
    return vec


def get_predictions(net, test_vids, category_names):
    fc3d = h5py.File(C3D, 'r')
    fpca = h5py.File(C3D_PCA, 'r')
    train_mean = get_training_mean(MEANFILE)
    pred = {}
    c3d_lev2 = pd.DataFrame(np.zeros((len(test_vids), len(category_names))), \
                            index=test_vids, columns=category_names)
    for i, vid in enumerate(test_vids):
        print "{} --> For video : {}" .format(i, vid)
        (rows, cols) = fc3d['v_'+vid]['c3d_features'].shape
        vid_data = fc3d['v_'+vid]['c3d_features'][:]
        # get predictions for each row of c3d feature
        vid_probs = pd.DataFrame(np.zeros((rows, len(category_names))), \
                                 columns=category_names)
        predicted_labels = []
        for row in range(rows):
            #print "Dims of vid_data[row,:] = {}" .format(vid_data[row,:].shape)
            #print "Values = {}" .format(vid_data[row,:])
            f = vid_data[row,:].reshape(cols, 1, 1)
            #print "Values = {}" .format(f)
            f = f - train_mean
            #pr = net.forward()
            out = net.forward_all(data = np.asarray([f]))
            predicted_labels.append(out['prob'][0].argmax(axis=0))
            #print "Predicted Label : {} :: Name : {}" .format(predicted_label, category_names[predicted_label])
            #print "Rows :: "
            vid_probs.iloc[row,:] = out['prob'][0]
            #print vid_probs.iloc[row,:]
        #break
        # returns a list of dict like [{'score': score, 'label':labels[idx]}...]
#        fpred_lst = open('dumps/predictions/v_'+vid+'.txt', 'w')
#        for plabel in predicted_labels:
#            fpred_lst.write("%s, " % plabel)
#        fpred_lst.close()
#        with open("dumps/predictions/v_"+vid+".pkl", 'wb') as fp:
#            cPickle.dump(vid_probs, fp)
#        print "Files written !!"
        pred[vid], vprobs = globalPrediction(vid, category_names, vid_probs, predicted_labels)
        c3d_lev2.loc[vid,:] = vprobs
        #if i==10:
        #    break
    fc3d.close()
    fpca.close()
    return pred, c3d_lev2

def globalPrediction(vid, category_names, vid_probs, predicted_labels):
    """
        Get a matrix of probabilities over the classes for the c3d features of 
        a video. Generate the top 3 predictions from the prob matrix
    """
    anno_list = []
    # Idea 1 : To form the hist over the categories, each bin has sum of probs
    vprobs_sum = vid_probs.sum(axis=0)
    top_n = vprobs_sum.sort_values(ascending = False)[:3]
    #counter = collections.Counter(predicted_labels)
    #top_n = counter.most_common(3)      # list of tuples
    #assert len(top_n)==3
    labels = top_n.index.tolist()
    scores = top_n.values.tolist()
    for idx,score in enumerate(scores):
        anno_list.append({'score': score, 'label':labels[idx]})
    #for (idx,score) in top_n:
    #    anno_list.append({'score': score, 'label':category_names[idx]})
    
    # Idea 2 : Detect temporal continuity of category predicted. Longer the better
    
    # Idea 3 : Count the number of highest votes for top category. (Worse than 1)
    # If equal votes for >1 category then use Idea 1
    # finds the max val index among the columns for each row and the freq of the 
    # occurrence of the column names (in decreasing order)
#    labels = vid_probs.idxmax(axis=1).value_counts()[:3].index.tolist()
#    scores = probs_sum[labels].tolist()
#    for idx,score in enumerate(scores):
#        anno_list.append({'score': score, 'label':labels[idx]})
    
    return anno_list, vprobs_sum
    

def get_temporalProps(net, test_vids, meta_info, category_names, n):
    fc3d = h5py.File(C3D, 'r')
    fpca = h5py.File(C3D_PCA, 'r')
    train_mean = get_training_mean(MEANFILE)
    pred = {}
    c3d_lev2 = pd.DataFrame(np.zeros((len(test_vids), len(category_names))), \
                            index=test_vids, columns=category_names)
    for i, vid in enumerate(test_vids):
        print "{} --> For video : {}" .format(i, vid)
        (rows, cols) = fc3d['v_'+vid]['c3d_features'].shape
        vid_data = fc3d['v_'+vid]['c3d_features'][:]
        # get predictions for each row of c3d feature of vid
        vid_probs = pd.DataFrame(np.zeros((rows, len(category_names))), \
                                 columns=category_names)
        predicted_labels = []
###############################################################################        
#        with open('dumps/predictions/v_'+vid+'.txt', 'r') as flst:
#            plabel = flst.read().split(',')
#        plabel.pop()        # remove last element ' '
#        plabel = map(int, plabel)
#        with open('dumps/predictions/v_'+vid+'.pkl', 'rb') as fp:
#            vid_probs = cPickle.load(fp)
#        print "Files read. {} !!" .format(plabel[:5])
#        (rows, cols) = vid_probs.shape
###############################################################################
        for row in range(rows):
            #print "Dims of vid_data[row,:] = {}" .format(vid_data[row,:].shape)
            #print "Values = {}" .format(vid_data[row,:])
            f = vid_data[row,:].reshape(cols, 1, 1)
            #print "Values = {}" .format(f)
            f = f - train_mean
            #pr = net.forward()
            out = net.forward_all(data = np.asarray([f]))
            predicted_labels.append(out['prob'][0].argmax(axis=0))
            #print "Predicted Label : {} :: Name : {}" .format(predicted_label, category_names[predicted_label])
            #print "Rows :: "
            vid_probs.iloc[row,:] = out['prob'][0]
            #print vid_probs.iloc[row,:]
###############################################################################
           # predicted_labels.append(vid_probs.iloc[row,:].argmax(axis=0))
###############################################################################
        # returns a list of dict like [{'score': score, 'label':labels[idx]}...]
#        fpred_lst = open('dumps/predictions/v_'+vid+'.txt', 'w')
#        for plabel in predicted_labels:
#            fpred_lst.write("%s, " % plabel)
#        fpred_lst.close()
#        with open("dumps/predictions/v_"+vid+".pkl", 'wb') as fp:
#            cPickle.dump(vid_probs, fp)
        
        # For Proposal
#        pred[vid], vprobs = get_vidProposal(vid, vid_probs, plabel,\
#                            meta_info, category_names, n)
        # For localization
        import localization as loc
        pred[vid], vprobs = loc.get_vidLocalization(vid, vid_probs, predicted_labels, \
                            meta_info, category_names, n)
        c3d_lev2.loc[vid,:] = vprobs
        #break
    fc3d.close()
    fpca.close()
    return pred, c3d_lev2

def get_vidProposal(vid, vid_probs, vid_preds, meta_info, category_names, n):
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
        segments = get_segments_for_cat(vid_preds, topn_idx[idx], idx) 
        ##### get time in sec from video info
        if len(segments)>0:
            nSegs = len(segments)
            times = 3
            for i,(beg,end) in enumerate(segments):
                begtime = (beg+1)*8./fps
                endtime = (end+1)*8./fps
                if i > nSegs/3 and i < 2*nSegs/3:
                    times = 2
                elif i>=2*nSegs/3:
                    times = 1
                # taking score as the temporal extent of the activity of interest
                for t in range(times):
                    bt = begtime + np.random.randn(1)[0]
                    et = endtime + np.random.randn(1)[0]
                    if bt<0:
                        bt = 0
                        #if et > duration
                    anno_list.append({'score': end-beg, 'segment': [bt, et]})
    
    # Find the top predicted label
    return anno_list, vprobs_sum

def get_segments_for_cat(pred_lst, cat_id, nth_val):
    """Retrieve segments corresponding to category number 'cat_id' from the list of 
    category predictions 'pred_lst'. Return a list of segment tuples
    """
    int_seg_dist = 60    # 2 for 8*i frames
    seg_len_th = 3 - nth_val
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
    merged_segments, new_segments = [], []
    if len(segments)==0:
        return []
    (bPrev, ePrev) = segments[0]
    # Merge 'close' segments based on int_seg_dist
    for i,(bCurr,eCurr) in enumerate(segments):
        if i==0:
            continue
        if (bCurr-ePrev)>int_seg_dist :
            merged_segments.append((bPrev, ePrev))
            bPrev = bCurr
        ePrev = eCurr
    merged_segments.append((bPrev, ePrev))
    # Create a dict of segment lengths for each tuple in segments
    seg_lens = {}
    for idx,seg in enumerate(merged_segments):
        seg_lens[idx] = seg[1] - seg[0]
    # get segment idxs in decreasing order of lens (sort dict values and get keys)
    decr_seg_lens = sorted(seg_lens, key=seg_lens.get, reverse=True)
    # For very small length videos
    if len(pred_lst) < 3 or len(decr_seg_lens) < 3:
        seg_len_th = 0

    for idx in decr_seg_lens:
        if seg_lens[idx] < seg_len_th:
            break
        new_segments.append(merged_segments[idx])
        
    return new_segments

def get_training_mean(meanFilePath):
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open( meanFilePath , 'rb' ).read()
    blob.ParseFromString(data)
    arr = np.array( caffe.io.blobproto_to_array(blob) )
    out = arr[0]
    print "Shape : {} " .format(out.shape)
    print out
    return out
    #np.save( npyFilePath , out )

def generate_submission(pred, f_name):
    """Generate the submission file for any task: Untrimmed Classification
    pred: dictionary with vids and predictions
    f_name: filename prefix for the json file
    """
    out_dict = {'version':version}
    ext_data_dict = {'used': True, 'details': 'C3D features.'}    
    out_dict['results'] = pred
    out_dict['external_data'] = ext_data_dict
    json_filename = f_name+SUBSET+'.json'
    with open(json_filename, 'w') as fp:
        json.dump(out_dict, fp)


if __name__=='__main__':

    # Read the database, version and taxonomy from JSON file
    with open(ANNOTATION_FILE, "r") as fobj:
        data = json.load(fobj)

    database = data["database"]
    taxonomy = data["taxonomy"]
    version = data["version"]
    
    non_existing_videos = utils.crosscheck_videos(VIDEOPATH, ANNOTATION_FILE)

    print "No of non-existing videos: %d" % len(non_existing_videos)
    
    train_vids_all = []
    [train_vids_all.append(x) for x in database if database[x]['subset']=='training']
    # Find list of available training videos
    train_existing_vids = list(set(train_vids_all) - set(non_existing_videos))
    
    val_vids_all = []
    [val_vids_all.append(x) for x in database if database[x]['subset']==SUBSET]
    # Find list of available training videos
    val_existing_vids = list(set(val_vids_all) - set(non_existing_videos))
    
    ###########################################################################
    # Get categories information from the database (Train+Validation sets)
    category = []
    for x in database:
        cc = []
        for l in database[x]["annotations"]:
            cc.append(l["label"])
        category.extend(list(set(cc)))
    category_count = collections.Counter(category)

    category_names = sorted(category_count.keys())
    print "Total No of classes: %d" % len(category_names)
    
    #print category_names
    ###########################################################################
    # MBH and ImageNetShuffle Features in training_model_m2.py
    ###########################################################################
    # C3D features
    # Read the meta_info and sample_positions files
    samples_csv = "samples.csv"
    samples_val_csv = "samples_val.csv"
    with open("training_data_meta_info.json", "r") as fobj:
        meta_info = json.load(fobj)
    #construct_dataset(meta_info, samples_csv, category_names)
    
    with open("test_data_meta_info.json", "r") as fobj:
        val_meta_info = json.load(fobj)
    #construct_dataset(val_meta_info, samples_val_csv, category_names)
    
    
    ###########################################################################
    # Consider Taxonomy of the classes
    # Temporal Proposals
    
    ###########################################################################

    caffe.set_mode_gpu()
    # load the model
    
    bgThresholds = [105000, 115000]
    net = caffe.Net(MODEL, PRETRAINED, caffe.TEST)

    # Predict on the validation set videos for each value of bgThreshold
    #for th in bgThresholds:
    #pred, c3d_probs = get_predictions(net, val_vids_all, category_names)
    
    # For Task 3 : Temporal Proposals
#    for n in [3,4]:
#        #if n==0 or n==11:
#        #    continue
#        pred, c3d_probs = get_temporalProps(net, val_vids_all, val_meta_info, category_names, n+1)
#        generate_submission(pred, 'submission_t'+str(n+1)+'_frProp_')
#    
##############################################################################

    # Task 4 : Localization
    for n in [0]:
        pred, c3d_probs = get_temporalProps(net, val_vids_all, val_meta_info, category_names, n+1)
        generate_submission(pred, 'submission_t'+str(n+1)+'_frLocal_')
    
#
##############################################################################

# Use LMDB to get the predictions
    # MEANFILE is the path to the training mean binaryproto file    
#    train_mean = get_training_mean(MEANFILE)
#    print "Mean file : {}" .format(train_mean)
#    import lmdb
#    lmdb_env = lmdb.open(LMDB_FOLDER+'/val_c3d_lmdb')
#    lmdb_txn = lmdb_env.begin()
#    lmdb_cursor = lmdb_txn.cursor()
#    count = 0
#    correct = 0
#    for key, value in lmdb_cursor:
#        print "Count:"
#        print count
#        count = count + 1
#        datum = caffe.proto.caffe_pb2.Datum()
#        datum.ParseFromString(value)
#        label = int(datum.label)
#        image = caffe.io.datum_to_array(datum)
#        print "Shape 1 : {}" .format(image.shape)
#        #image = image.astype(np.uint8)
#        image = image - train_mean
#        print "Shape 2 : {}" .format(image.shape)
#        print "Asarray shape : {}" .format(np.asarray([image]).shape)
#        out = net.forward_all(data=np.asarray([image]))
#        print "out Shape : {}" .format(out['prob'].shape)
#        predicted_label = out['prob'][0].argmax(axis=0)
#        print "Predicted Label : {}" .format(predicted_label)
#        
#        if count == 3:
#            break

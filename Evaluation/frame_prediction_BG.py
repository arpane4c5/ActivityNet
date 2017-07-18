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
import cv2
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
MEANFILE = "/home/arpan/DATA_Drive/ActivityNet/ActivityNet-master/caffe_models/mean_c3d_4k.binaryproto"
MEANFILE = "/home/hadoop/VisionWorkspace/ActivityNet/ActivityNet-master/caffe_models/mean_c3d_4k.binaryproto"
SUBSET = 'validation'


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
    bgThresh = 500000
    
    print "Calculate frames being ignored ..."
    result = Parallel(n_jobs=4)(delayed(get_rows_ignored) \
                          (test_vids[j], bgThresh, j) \
                          for j in range(len(test_vids)))
    
    for i, vid in enumerate(test_vids):
        print "{} --> For video : {}" .format(i, vid)
        vid_data = fc3d['v_'+vid]['c3d_features'][:]
        frms_ignored = result[i]
        # get the c3d features that need to be ignored. Note that c3d features
        # are sampled every 8 frames, therefore position is divided by 8
        rows_ignored = [int(r/8) for r in frms_ignored]
        print "Rows ignored : {}" .format(set(rows_ignored))
        not_rows_ig = list(set(range(vid_data.shape[0])) - set(rows_ignored))
        (rows, cols) = vid_data.shape
        # get predictions for each row of c3d feature
        vid_probs = pd.DataFrame(np.zeros((rows, len(category_names))), \
                                 columns=category_names)
        #print frms_ignored
        for row in not_rows_ig:
            #print "Dims of vid_data[row,:] = {}" .format(vid_data[row,:].shape)
            #print "Values = {}" .format(vid_data[row,:])
            f = vid_data[row,:].reshape(cols, 1, 1)
            # Subtract mean
            f = f - train_mean
            out = net.forward_all(data = np.asarray([f]))
            predicted_label = out['prob'][0].argmax(axis=0)
            #print "Predicted Label : {} :: Name : {}" .format(predicted_label, category_names[predicted_label])
            #print "Rows :: "
            vid_probs.iloc[row,:] = out['prob'][0]
            #print vid_probs.iloc[row,:]
        # returns a list of dict like [{'score': score, 'label':labels[idx]}...]
        pred[vid], vprobs = globalPrediction(vid, category_names, vid_probs)
        print pred[vid]
        c3d_lev2.loc[vid,:] = vprobs
        #break
    fc3d.close()
    fpca.close()
    return pred, c3d_lev2
    

def globalPrediction(vid, category_names, vid_probs):
    """
        Get a matrix of probabilities over the classes for the c3d features of 
        a video. Generate the top 3 predictions from the prob matrix
    """
    anno_list = []
    # Idea 1 : To form the hist over the categories, each bin has sum of probs
    vprobs_sum = vid_probs.sum(axis=0)
    top_n = vprobs_sum.sort_values(ascending = False)[:3]
    labels = top_n.index.tolist()
    scores = top_n.values.tolist()
    for idx,score in enumerate(scores):
        anno_list.append({'score': score, 'label':labels[idx]})
        
    
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
    
def get_rows_ignored(vid, bgThresh, v_no):
    """
    Use background subtraction to decide which frames to ignore while prediction
    """
    # process the video frame by frame
    print "For video : {} " .format(v_no)
    W, H = 160, 120
    vpath = os.path.join(VIDEOPATH, 'v_'+vid+'.mp4')
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        raise IOError("Capture object not opened !")
    #fps = cap.get(cv2.CAP_PROP_FPS)
    frms_ig = []
    frms_msec = []
    fgbg = cv2.createBackgroundSubtractorMOG2()     #bg subtractor
    ret, prev_frame = cap.read()
    prev_frame = cv2.resize(prev_frame, (W, H) )
    fgmask = fgbg.apply(prev_frame)
    # convert frame to GRAYSCALE
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    # iterate over the frames 
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (W, H))
        curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # To find the background mask and skip the frame if foreground is absent
        fgmask = fgbg.apply(frame)
        if np.sum(fgmask)<bgThresh:
            #print "BG frame skipped !!"
            #print "FGMASK : {}" .format(np.sum(fgmask))
            frms_ig.append(cap.get(cv2.CAP_PROP_POS_FRAMES))
            frms_msec.append(cap.get(cv2.CAP_PROP_POS_MSEC))
            count += 1
            #cv2.imshow("BG Ignored", curr_frame)
            #waitTillEscPressed()
            prev_frame = curr_frame
            continue

    #print "Total Frames : {}" .format(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #print "Skipped Frames : {}" .format(count)
    #print frms_ig
    #print frms_msec
    cap.release()
    #cv2.destroyAllWindows()
    return frms_ig

def waitTillEscPressed():
    while(True):
        # For moving forward
        if cv2.waitKey(10)==27:
            print("Esc Pressed. Move Forward without labeling.")
            return 1

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
    
    with open("val_data_meta_info.json", "r") as fobj:
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
    pred, c3d_probs = get_predictions(net, val_existing_vids, category_names)
    
    print "Predicted Labels : "
    print c3d_probs.head()

#
    out_dict = {'version':version}
    subset_video_ids = []
    ext_data_dict = {'used': True, 'details': 'C3D features.'}
    
    out_dict['results'] = pred
    out_dict['external_data'] = ext_data_dict
            
    json_filename = 'submission_t3_framewise_'+SUBSET+'.json'
    with open(json_filename, 'w') as fp:
        json.dump(out_dict, fp)
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

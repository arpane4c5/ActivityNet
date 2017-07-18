#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 22:56:22 2017

@author: Arpan

Description: Use frame HOG features
"""

import json
import os
import utils
import numpy as np
import h5py
import pandas as pd
import collections
import cPickle
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed
import lmdb
import caffe
import cv2


# Temporal Proposals : Pretrained
#VIDEOPATH = '/home/arpan/DATA_Drive/ActivityNet/videos'
#ANNOTATION_FILE = '/home/arpan/DATA_Drive/ActivityNet/ActivityNet-master/Evaluation/data/activity_net.v1-3.min.json'
#PROPOSALS_FILENAME = '/home/arpan/DATA_Drive/ActivityNet/extra_features/Temporal Activity Proposals/activitynet_v1-3_proposals.hdf5'
#SHUFFLE = '/home/arpan/DATA_Drive/ActivityNet/extra_features/ImageNet Shuffle Features/ImageNetShuffle2016_features.h5'
#MBH = "/home/arpan/VisionWorkspace/ActivityNet/MBH Features/MBH_Videos_features.h5"
#MBH_IDS = "/home/arpan/VisionWorkspace/ActivityNet/MBH Features/MBH_Videos_quids.txt"
#C3D = "/home/arpan/DATA_Drive/ActivityNet/extra_features/C3D Features/sub_activitynet_v1-3.c3d.hdf5"
#C3D_PCA = "/home/arpan/DATA_Drive/ActivityNet/extra_features/C3D Features/PCA_activitynet_v1-3.hdf5"
#SHUFFLE_IDS = '/home/arpan/DATA_Drive/ActivityNet/extra_features/ImageNet Shuffle Features/ImageNetShuffle2016_quids.txt'
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
HOGFILE = "/home/hadoop/VisionWorkspace/ActivityNet/ActivityNet-master/Evaluation/hog.xml"
SUBSET = 'validation'

def construct_dataset(meta_info, samples_csv, category_names, prefix):
    
    lmdb_name = os.path.join(LMDB_FOLDER, prefix+"_hog_lmdb")
    if not os.path.exists(os.path.dirname(lmdb_name)):
        os.makedirs(os.path.dirname(lmdb_name))
    
    samples_df = pd.read_csv(samples_csv)
    print "Creating HOG features..."
    
    # Create lmdb
    (H, W, C) = (1, 1, 2000)
    N = samples_df.shape[0]     # no of rows (=no of visualizations = 5k)
    # twice the size of total number of OF visualizations
    map_size = int(N*H*W*C*3*15)     # approx 429 GB
    #map_size = int(N*720*1280*C*2)     # approx 429 GB
    
    env = lmdb.open(lmdb_name, map_size=map_size)    
    i = 0   # LMDB index variable
    # iterate over the rows of the pandas dataframe
    end_samples = samples_df.shape[0]
    r = (end_samples - i)/200
    print "No of samples per class = %d " %r
    ###########################################################################
    nCat = 4*len(category_names)          # = 800 per batch
    nCat_samples = (end_samples - i)/nCat    # = N = 1000
    lmdb_id = 0
    
    # HOG returns a 9576 sized vector
    # Parallelizing the lmdb creation process
    for i in range(nCat_samples):
        
        result = Parallel(n_jobs=4)(delayed(get_hog_feature) \
                          (samples_df['video_id'][i*nCat+j], \
                           samples_df['position'][i*nCat+j])
                          for j in range(nCat))
        
        with env.begin(write = True) as txn:
            for l,vec in enumerate(result):
                row_no = (i*nCat)+l
                pos = samples_df['position'][row_no]
                video_id = samples_df['video_id'][row_no]
                lab = samples_df['label'][row_no]
                print "idx : "+str(row_no)+" :: 'position' : "+str(pos)
                
                #img = np.rollaxis(img, 2)   # C, H, W
                datum = caffe.proto.caffe_pb2.Datum()
                # since it is a vector, it only has 1st dimension
                #print "vec shape : {}" .format(vec.shape)
                datum.channels = vec.shape[0]
                datum.height = 1
                datum.width = 1
                #datum.data = img.tobytes()
                datum.float_data.extend(vec.astype(float).flat)
                datum.label = lab
                str_id = '{:08}'.format(lmdb_id)
                # The encode is only essential in Python 3
                txn.put(str_id.encode('ascii'), datum.SerializeToString())
                lmdb_id += 1
        print "Write No : %d" %(i+1)
    print "LMDB construction successful !"
    return

def get_hog_feature(vid, pos):
    '''
    Read the frame at 'pos' of video and find the hog feature for the frame
    '''
    height, width = 120, 160
    cap = cv2.VideoCapture(os.path.join(VIDEOPATH, 'v_'+vid+'.mp4'))
    if not cap.isOpened():
        raise IOError('Capture object not opened !')
    hog = cv2.HOGDescriptor("hog.xml")
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    ret, frame = cap.read()
    while not ret:
        print "Frame not read. Move backwards."
        pos -= 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
    
    frame = cv2.resize(frame, (width, height))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("frame", frame)
    #waitTillEscPressed()
    hist = hog.compute(frame)
    cols = hist.shape[0]
    #print "HOG Shape : {}" .format(hist.shape)
    #print "Reshaped : {}" .format(hist.reshape((cols)).shape)
    hist = hist.reshape((cols))
    cap.release()
    #cv2.destroyAllWindows()
    return hist
    
def waitTillEscPressed():
    while(True):
        # For moving forward
        if cv2.waitKey(10)==27:
            print("Esc Pressed. Move Forward without labeling.")
            return 1


def predict_on_glFeat(X_val, nFeatures, database, category_names, val_existing_vids, \
                      destPath, nEstimators):
    
    # Create a dataframe with rows as egs and cols as class 1 prob values
    threshold = 0.5
    X = X_val.loc[:,range(nFeatures)]
    y_prob = pd.DataFrame(np.zeros((len(X_val), len(category_names))), \
                          columns=category_names, index=X_val.index)
    for cat in category_names:
        # load the model
        f_name = os.path.join(destPath+"_"+str(nEstimators),\
                              destPath+"_"+str(nEstimators)+"_"+cat+".pkl")
        with open(f_name, "rb") as fid:
            rf_model = cPickle.load(fid)
        
        # Assign positive class probabilities 
        y_prob[cat] = rf_model.predict_proba(X)[:,1]
        print "No. of examples above threshold for class {} : {}" \
                        .format(cat, sum(y_prob[cat]>threshold))
    
    # Top 5 predictions
    pred = {}
    #y_prob.apply(np.argmax, axis=1)
    for vid in list(X.index):
        #print "ID : %s " %vid
        # select top 3 prediction values and their labels and save in dict
        top_n = y_prob.loc[vid,:].sort_values(ascending=False)[:3]
        labels = top_n.index.tolist()
        scores = top_n.values.tolist()
        pred[vid] = []
        for idx,score in enumerate(scores):
            pred[vid].append({'score': score, 'label':labels[idx]})
        
    return pred, y_prob


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
    # Create HOG feature dataset
     
    # Read the meta_info and sample_positions files
    samples_csv = "tr_samples_4k.csv"
    samples_val_csv = "val_samples_1k.csv"
    with open("training_data_meta_info.json", "r") as fobj:
        meta_info = json.load(fobj)
    construct_dataset(meta_info, samples_csv, category_names, "test_train")
    
    with open("val_data_meta_info.json", "r") as fobj:
        val_meta_info = json.load(fobj)
    construct_dataset(val_meta_info, samples_val_csv, category_names, "test_val")
    
    # train a model without convolution layers, only fc layers should be there
    
    
    ###########################################################################
    # Consider Taxonomy of the classes
    # Temporal Proposals
    
    ###########################################################################
    
#    out_dict = {'version':version}
#    subset_video_ids = []
#    ext_data_dict = {'used': True, 'details': \
#                'C3D features.'}
#    
#    out_dict['results'] = pred
#    out_dict['external_data'] = ext_data_dict
#            
#    json_filename = 'submission_t3_'+SUBSET+'.json'
#    with open(json_filename, 'w') as fp:
#        json.dump(out_dict, fp)
    ###########################################################################

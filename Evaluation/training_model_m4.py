#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 22:56:22 2017

@author: Arpan

Description: Use  C3D features
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
SUBSET = 'validation'

def construct_dataset(meta_info, samples_csv, category_names, prefix):
    
    lmdb_name = os.path.join(LMDB_FOLDER, prefix+"_c3d_lmdb")
    if not os.path.exists(os.path.dirname(lmdb_name)):
        os.makedirs(os.path.dirname(lmdb_name))
    
    samples_df = pd.read_csv(samples_csv)
    print "Loading C3D features..."
    fc3d = h5py.File(C3D, 'r')
    fpca = h5py.File(C3D_PCA, 'r')
    
    # Create lmdb
    (H, W, C) = (1, 1, 500)
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
    nCat = 4*len(category_names)          # = 200
    nCat_samples = (end_samples - i)/nCat    # = N = 1000
    lmdb_id = 0
                   
    # Parallelizing the lmdb creation process
    for i in range(nCat_samples):
        
        result = Parallel(n_jobs=1)(delayed(get_c3d_feature) \
                          (fc3d, 'v_'+samples_df['video_id'][i*nCat+j], \
                           samples_df['position'][i*nCat+j], \
                        meta_info[samples_df['video_id'][i*nCat+j]]['fps']) \
                          for j in range(nCat))
        
        with env.begin(write = True) as txn:
            for l in range(len(result)):
                row_no = (i*nCat)+l
                pos = samples_df['position'][row_no]
                video_id = samples_df['video_id'][row_no]
                lab = samples_df['label'][row_no]
                print "idx : "+str(row_no)+" :: 'position' : "+str(pos)
                
                for vec in result[l]:
                    #img = np.rollaxis(img, 2)   # C, H, W
                    datum = caffe.proto.caffe_pb2.Datum()
                    # since it is a vector, it only has 1st dimension
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
    fc3d.close()
    fpca.close()
    return

def get_c3d_feature(fc3d, vid, pos, vfps):
    '''
    Read the feature vector that is near the pos of video
    c3d features are taken for every 8th frame
    '''
    vec = []
    #print "vid : {} :: pos : {} :: vfps : {}" .format(vid, pos, vfps)
    #print "Shape : {}" .format(fc3d[vid]['c3d_features'].shape)
    row = int(pos/8)
    while not row < fc3d[vid]['c3d_features'].shape[0]:
        #print "Decrement by 1"
        row -= 1
    vec.append(fc3d[vid]['c3d_features'][row,:])
    return vec
    

def partition_dataset(feature, train_vids_all, val_existing_vids):
    if feature == "C3D":
        print "Loading C3D features..."
        fc3d = h5py.File(C3D, 'r')
        fpca = h5py.File(C3D_PCA, 'r')        
    else:
        raise IOError("Invalid first argument: "+feature)

    for vid in fobj.keys():
        fc3d[vid]['c3d_features'][:]
    # Too large, need >10GB memory, for MBH
    X_all = fobj['features'][:]
    X_all = pd.DataFrame(X_all , index=ids)
    X_train = X_all.loc[train_vids_all]
    X_val = X_all.loc[val_existing_vids]
    del X_all
    fobj.close()
    print "X_train = {} " .format(X_train.shape)
    nFeat = X_train.shape[1]
    y_train = pd.DataFrame(np.zeros((len(X_train), len(category_names))),\
                           columns=category_names, index=X_train.index)
    y_val = pd.DataFrame(np.zeros((len(X_val), len(category_names))), \
                         columns=category_names, index=X_val.index)
    
    # Join the columns for each category
    X_train = pd.concat([X_train, y_train], axis = 1)
    X_val = pd.concat([X_val, y_val], axis = 1)
    #print X_train.head()
    # Iterate over the videos of X_train and X_val and set labels
    for vid in train_vids_all:
        for annotation in database[vid]['annotations']:
            X_train.at[vid, annotation['label']] = 1
    
    print "Labels set for Training Set !"

    for vid in val_existing_vids:
        for annotation in database[vid]['annotations']:
            X_val.at[vid, annotation['label']] = 1
    
    print "Labels set for Validation Test !"

    return X_train, X_val, nFeat


def train_on_glFeat(X_train, nFeatures, database, category_names, train_vids_all, \
                    destPath, seed, nEstimators):
    """Function to read the MBH features and train a classifier for each class.
    Input: 
    feature: "MBH" for training on MBH features and "SHUFFLE" for training on shuffle
    database: read from JSON file
    category_names: sorted list of class names
    train_vids_all: list of video ids in the training set
    nEstimators: no of trees for Random Forest
    """
    #print X_train.head()
    	# Iterate over the categories and for each category, prepare the dataset
    for cat in category_names:
        # for a cat, find the video IDs which have labels
        pos_samples = X_train[X_train[cat]==1]
        pos_samples = pos_samples.loc[:, range(nFeatures)+[cat]]
        # sample negative rows equal to the no of pos examples
        neg_samples = X_train[X_train[cat]==0].sample(n=len(pos_samples), \
                             random_state = 321)
        neg_samples = neg_samples.loc[:, range(nFeatures)+[cat]]
        
        # join pos and negative samples
        X = pd.concat([pos_samples, neg_samples])
        
        X = X.sample(frac=1, random_state=231)    # shuffle
        y = np.array(X[cat])
        X = X.loc[:,range(nFeatures)]
        
        rf_model = train_model_rf(X, y, estimators = nEstimators, seed=seed)
        
        if not os.path.exists(destPath+"_"+str(nEstimators)):
            os.makedirs(destPath+"_"+str(nEstimators))
        f_name = os.path.join(destPath+"_"+str(nEstimators), \
                              destPath+"_"+str(nEstimators)+"_"+cat+".pkl")
        with open(f_name, "wb") as fid:
            cPickle.dump(rf_model, fid)
        print "Model saved for category : %s " %cat

    print "Models Trained and saved to files."
    # this returns a list of 10 SVM
    #result = Parallel(n_jobs=3)(delayed(train_model_rf)(X, y, seed) for seed in range(10))
    
def train_model_rf(X, y, estimators, seed):
    # select the parameters, generate probabilities etc
    clf = RandomForestClassifier(n_estimators = estimators, random_state=seed)
    clf = clf.fit(X, y)
    return clf

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


def train_on_C3D(database, category_names, train_vids_all):
    """Function to read the C3D features and train a model on them
    """


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
    samples_csv = "tr_samples_10k.csv"
    samples_val_csv = "val_samples_2500.csv"
    with open("training_data_meta_info.json", "r") as fobj:
        meta_info = json.load(fobj)
    construct_dataset(meta_info, samples_csv, category_names, "train")
    
    with open("val_data_meta_info.json", "r") as fobj:
        val_meta_info = json.load(fobj)
    construct_dataset(val_meta_info, samples_val_csv, category_names, "val")
    
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

    


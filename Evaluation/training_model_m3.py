#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 20:53:14 2017

@author: Arpan

Description: Training Model : Method 2
Using CNNs for training. Pretrained models
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


# Temporal Proposals : Pretrained
VIDEOPATH = '/home/hadoop/VisionWorkspace/ActivityNet/ActivityNet-master/Crawler/videos'
ANNOTATION_FILE = '/home/hadoop/VisionWorkspace/ActivityNet/ActivityNet-master/Evaluation/data/activity_net.v1-3.min.json'
PROPOSALS_FILENAME = '/home/hadoop/VisionWorkspace/ActivityNet/Downloads/Temporal Activity Proposals/activitynet_v1-3_proposals.hdf5'
SHUFFLE = '/home/hadoop/VisionWorkspace/ActivityNet/Downloads/ImageNet Shuffle Features/ImageNetShuffle2016_features.h5'
MBH = "/home/hadoop/VisionWorkspace/ActivityNet/Downloads/MBH Features/MBH_Videos_features.h5"
C3D = "/home/hadoop/VisionWorkspace/ActivityNet/Downloads/C3D Features/sub_activitynet_v1-3.c3d.hdf5"
C3D_PCA = "/home/hadoop/VisionWorkspace/ActivityNet/Downloads/C3D Features/PCA_activitynet_v1-3.hdf5"
SHUFFLE_IDS = '/home/hadoop/VisionWorkspace/ActivityNet/Downloads/ImageNet Shuffle Features/ImageNetShuffle2016_quids.txt'
SUBSET = 'validation'


def train_model_linSVM(X, y):
    # select the parameters, generate probabilities etc
    model = svm.LinearSVC()
    return model.fit(X, y)


def train_on_shuffle(database, category_names, train_vids_all, destPath="shuffle_RF"):
    
    # ImageNet shuffle features:
    # 19994 x 1024 features
    fobj = h5py.File(SHUFFLE, 'r')
    
    # shape is 19994 x 1024
    print "Shape : {}" .format(fobj['features'].shape)
    
    # As the videos are sorted, the index created will be the video_no
    # corresponding to the video row in h5 database.
    shuffle_ids = pd.read_csv(SHUFFLE_IDS, header='infer', \
                              names = ['id'], usecols = [2])
    
    sh_id = [s.split('_', 1)[-1] for s in shuffle_ids['id']]
    sh_id = [s.rsplit('.',1)[0] for s in sh_id]
    # Copy all the values to numpy array var
    X_all = fobj['features'][:]
    
    # join features with video_ids
    X_all = pd.DataFrame(X_all, index=sh_id)
    
    # subset rows for training and validation
    X_train = X_all.loc[train_vids_all]
    #X_val = X_all[X_all['id'].isin(val_vids_appended)]
    del X_all
    fobj.close()
    print "X_train = {} " .format(X_train.shape)
    y_train = pd.DataFrame(np.zeros((len(X_train), len(category_names))),\
                           columns=category_names, index=X_train.index)
    
    # Join the columns for each category
    X_train = pd.concat([X_train, y_train], axis = 1)

    #print X_train.head()
    # Iterate over the videos of X_train and X_val and set labels
    for vid in train_vids_all:
        for annotation in database[vid]['annotations']:
            X_train.at[vid, annotation['label']] = 1
    
    print "Labels set !"
    #print X_train.head()
    # Iterate over the categories and for each category, prepare the dataset
    for cat in category_names:
        # for a cat, find the video IDs which have labels
        pos_samples = X_train[X_train[cat]==1]
        pos_samples = pos_samples.loc[:, range(1024)+[cat]]
        # sample negative rows equal to the no of pos examples
        neg_samples = X_train[X_train[cat]==0].sample(n=len(pos_samples), \
                             random_state = 321)
        neg_samples = neg_samples.loc[:, range(1024)+[cat]]
        
        # join pos and negative samples and shuffle
        X = pd.concat([pos_samples, neg_samples])
        
        X = X.sample(frac=1, random_state=231)    # shuffle
        y = np.array(X[cat])
        X = X.loc[:,range(1024)]
        
        rf_model = train_model_rf(X, y, estimators=20, seed=123)
        
        if not os.path.exists(destPath):
            os.makedirs(destPath)
        f_name = os.path.join(destPath, destPath+"_"+cat+".pkl")
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

def predict_on_shuffle(database, category_names, val_existing_vids, destPath="RF"):
    # ImageNet shuffle features:
    # 19994 x 1024 features
    fobj = h5py.File(SHUFFLE, 'r')
    # shape is 19994 x 1024
    print "Shape : {}" .format(fobj['features'].shape)
    
    shuffle_ids = pd.read_csv(SHUFFLE_IDS, header='infer', \
                              names = ['id'], usecols = [2])
    
    sh_id = [s.split('_', 1)[-1] for s in shuffle_ids['id']]
    sh_id = [s.rsplit('.',1)[0] for s in sh_id]
    # Copy all the values to numpy array var
    X_all = fobj['features'][:]
    
    # join features with video_ids
    X_all = pd.DataFrame(X_all, index=sh_id)
    
    # subset rows for validation set
    X_val = X_all.loc[val_existing_vids]
    #X_val = X_all[X_all['id'].isin(val_vids_appended)]
    del X_all
    fobj.close()
    print "X_val = {} " .format( X_val.shape)
    y_val = pd.DataFrame(np.zeros((len(X_val), len(category_names))), \
                         columns=category_names, index=X_val.index)
    
    # Join the columns for each category
    X_val = pd.concat([X_val, y_val], axis = 1)
    
    for vid in val_existing_vids:
        for annotation in database[vid]['annotations']:
            X_val.at[vid, annotation['label']] = 1
    
    print "Labels set !"
    # Create a dataframe with rows are classes and cols are 0 class and 1 class
    # Probability values
    #prob = pd.DataFrame
    X = X_val.loc[:,range(1024)]
    y_prob = pd.DataFrame(np.zeros((len(X_val), len(category_names))), \
                          columns=category_names, index=X_val.index)
    for cat in category_names:
        # load the model
        f_name = os.path.join(destPath, destPath+"_"+cat+".pkl")
        with open(f_name, "rb") as fid:
            rf_model = cPickle.load(fid)
        
        # Assign positive class probabilities 
        y_prob[cat] = rf_model.predict_proba(X)[:,1]
        
        print "Probabilities for class {} : {}" .format(cat,y_prob[cat])
    
    # Top 5 predictions
    threshold = 0.5
    pred = {}
    #y_prob.apply(np.argmax, axis=1)
    for vid in list(X.index):
        #print "ID : %s " %vid
        # select top 5 prediction values and their labels and save in dict
        top_n = y_prob.loc[vid,:].sort_values(ascending=False)[:3]
        labels = top_n.index.tolist()
        scores = top_n.values.tolist()
        pred[vid] = []
        for idx,score in enumerate(scores):
            pred[vid].append({'score': score, 'label':labels[idx]})
        
    return pred
    
        
# for testing the functions
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
    
    
    
    # Temporal Proposals 
    # Optimized for high recall
    # 19994 x Mi proposals (For each video a number of proposals 
    # each with a score in decreasing order)

    # MBH
    # 19994 x 65536  features
    # 
    
    # ImageNet Shuffle Features
    #train_on_shuffle(database, category_names, train_vids_all, "RF")
    
    pred = predict_on_shuffle(database, category_names, val_existing_vids, "RF")
    out_dict = {'version':version}
    subset_video_ids = []
    ext_data_dict = {'used': False, 'details': \
                'Describe the external data over here. If necessary for each prediction'}
    
    out_dict['results'] = pred
    out_dict['external_data'] = ext_data_dict
            
    json_filename = 'submission_t3_'+SUBSET+'.json'
    with open(json_filename, 'w') as fp:
        json.dump(out_dict, fp)
    # Step 1: Form the datasets
    # To train 200 SVMs, each for an activity class.
    # Use One Vs All SVM ( for not used LinearSVC, which is a multi-class classifier )
    

    
    
    # training videos_info is in meta_info
    # check whether a particular video is 
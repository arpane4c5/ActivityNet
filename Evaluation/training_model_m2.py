#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 22:56:22 2017

@author: Arpan

Description: Use MBH Features and ImageNetShuffle Features
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
VIDEOPATH = '/home/arpan/DATA_Drive/ActivityNet/videos'
ANNOTATION_FILE = '/home/arpan/DATA_Drive/ActivityNet/ActivityNet-master/Evaluation/data/activity_net.v1-3.min.json'
PROPOSALS_FILENAME = '/home/arpan/DATA_Drive/ActivityNet/extra_features/Temporal Activity Proposals/activitynet_v1-3_proposals.hdf5'
SHUFFLE = '/home/arpan/DATA_Drive/ActivityNet/extra_features/ImageNet Shuffle Features/ImageNetShuffle2016_features.h5'
MBH = "/home/arpan/VisionWorkspace/ActivityNet/MBH Features/MBH_Videos_features.h5"
MBH_IDS = "/home/arpan/VisionWorkspace/ActivityNet/MBH Features/MBH_Videos_quids.txt"
C3D = "/home/arpan/DATA_Drive/ActivityNet/extra_features/C3D/sub_activitynet_v1-3.c3d.hdf5"
C3D_PCA = "/home/arpan/DATA_Drive/ActivityNet/extra_features/C3D/PCA_activitynet_v1-3.hdf5"
SHUFFLE_IDS = '/home/arpan/DATA_Drive/ActivityNet/extra_features/ImageNet Shuffle Features/ImageNetShuffle2016_quids.txt'
MODEL = "/home/arpan/DATA_Drive/ActivityNet/ActivityNet-master/caffe_models/deploy_c3d_fc_net.prototxt"
PRETRAINED = "/home/arpan/DATA_Drive/ActivityNet/ActivityNet-master/caffe_models/snapshots/c3d_4k_1k/c3d_fc_net_snap_iter_400000.caffemodel"
SUBSET = 'validation'

#VIDEOPATH = '/home/hadoop/VisionWorkspace/ActivityNet/ActivityNet-master/Crawler/videos'
#ANNOTATION_FILE = '/home/hadoop/VisionWorkspace/ActivityNet/ActivityNet-master/Evaluation/data/activity_net.v1-3.min.json'
#PROPOSALS_FILENAME = '/home/hadoop/VisionWorkspace/ActivityNet/Downloads/Temporal Activity Proposals/activitynet_v1-3_proposals.hdf5'
#SHUFFLE = '/home/hadoop/VisionWorkspace/ActivityNet/Downloads/ImageNet Shuffle Features/ImageNetShuffle2016_features.h5'
#MBH = "/home/hadoop/VisionWorkspace/ActivityNet/Downloads/MBH Features/MBH_Videos_features.h5"
#MBH_IDS = "/home/hadoop/VisionWorkspace/ActivityNet/Downloads/MBH Features/MBH_Videos_quids.txt"
#C3D = "/home/hadoop/VisionWorkspace/ActivityNet/Downloads/C3D Features/sub_activitynet_v1-3.c3d.hdf5"
#C3D_PCA = "/home/hadoop/VisionWorkspace/ActivityNet/Downloads/C3D Features/PCA_activitynet_v1-3.hdf5"
#SHUFFLE_IDS = '/home/hadoop/VisionWorkspace/ActivityNet/Downloads/ImageNet Shuffle Features/ImageNetShuffle2016_quids.txt'
#MODEL = "/home/hadoop/VisionWorkspace/ActivityNet/ActivityNet-master/caffe_models/deploy_c3d_fc_net.prototxt"
#PRETRAINED = "/home/hadoop/VisionWorkspace/ActivityNet/ActivityNet-master/caffe_models/snapshots/c3d_4k_1k/c3d_fc_net_snap_iter_400000.caffemodel"
#SUBSET = 'validation'

def partition_dataset(feature, train_vids_all, val_existing_vids):
    if feature == "MBH":
        print "Loading MBH features..."
        fobj = h5py.File(MBH, 'r')
        # As the videos are sorted, the index created will be the video_no
        # corresponding to the video row in h5 database.
        ids = pd.read_csv(MBH_IDS, header='infer', \
                          names = ['id'], usecols = [2])
    elif feature == "SHUFFLE":
        # ImageNet shuffle features:
        # 19994 x 1024 features
        print "Loading ImageNetShuffle features..."
        fobj = h5py.File(SHUFFLE, 'r')
        ids = pd.read_csv(SHUFFLE_IDS, header='infer', \
                                  names = ['id'], usecols = [2])
    else:
        raise IOError("Invalid first argument: "+feature)
    
    ids = [s.split('_', 1)[-1] for s in ids['id']]
    ids = [s.rsplit('.',1)[0] for s in ids]
    # Too large, need >10GB memory, for MBH
    X_all = fobj['features'][:]
    X_all = pd.DataFrame(X_all , index=ids)
    X_train = X_all.loc[train_vids_all]
    # normalize features
    X_train_norm = (X_train - X_train.mean())/(X_train.max() - X_train.min())
    X_val = X_all.loc[val_existing_vids]
    # use training mean, max and min values to transform X_val
    X_val_norm = (X_val - X_train.mean()) / (X_train.max() - X_train.min())
    del X_all, X_train, X_val
    fobj.close()
    print "X_train = {} " .format(X_train_norm.shape)
    nFeat = X_train_norm.shape[1]
    y_train = pd.DataFrame(np.zeros((len(X_train_norm), len(category_names))),\
                           columns=category_names, index=X_train_norm.index)
    y_val = pd.DataFrame(np.zeros((len(X_val_norm), len(category_names))), \
                         columns=category_names, index=X_val_norm.index)
    
    # Join the columns for each category
    X_train_norm = pd.concat([X_train_norm, y_train], axis = 1)
    X_val_norm = pd.concat([X_val_norm, y_val], axis = 1)
    #print X_train.head()
    # Iterate over the videos of X_train and X_val and set labels
    for vid in train_vids_all:
        for annotation in database[vid]['annotations']:
            X_train_norm.at[vid, annotation['label']] = 1
    
    print "Labels set for Training Set !"

    for vid in val_existing_vids:
        for annotation in database[vid]['annotations']:
            X_val_norm.at[vid, annotation['label']] = 1
    
    print "Labels set for Validation Test !"

    return X_train_norm, X_val_norm, nFeat


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
    
def train_level2_SVM( sh, c3d, y, destPath):
    if not os.path.exists(destPath):
        os.makedirs(destPath)
    
    for cat in y.columns.tolist():
        X_cat = pd.concat([ sh.loc[:,cat], c3d.loc[:,cat], y.loc[:,cat]], axis=1)
        # 'm_'+cat,
        X_cat.columns = [ 's_'+cat, 'c_'+cat, 'y_'+cat]
        pos_samples = X_cat[X_cat['y_'+cat]==1]
        # sample negative rows equal to the no of pos examples
        neg_samples = X_cat[X_cat['y_'+cat]==0].sample(n=len(pos_samples), \
                             random_state = 456)
        
        # join pos and negative samples
        X = pd.concat([pos_samples, neg_samples])
        X = X.sample(frac=1, random_state=654)    # shuffle
        
        # 'm_'+cat,
        svm_model = train_model_linSVM(X.loc[:,['s_'+cat, 'c_'+cat]],\
                                       np.array(X.loc[:,'y_'+cat]))
        
        f_name = os.path.join(destPath, 'svm_'+cat+'.pkl')
        with open(f_name, 'wb') as fid:
            cPickle.dump(svm_model, fid)
    print "SVM models saved!"

def train_model_linSVM(X, y):
    # select the parameters, generate probabilities etc
    model = svm.SVC(kernel = 'linear', probability = True)
    return model.fit(X, y)

def predict_on_SVM( p_sh, p_c3d, destPath):
    threshold = 0.5
    category_names = p_sh.columns.tolist()
    y_prob = pd.DataFrame(np.zeros((len(p_sh), len(category_names))), \
                          columns=category_names, index=p_sh.index)
    for cat in category_names:
        # p_mbh[cat],
        X = pd.concat([ p_sh[cat], p_c3d[cat]], axis = 1)
        # 'm_'+cat,
        X.columns = [ 's_'+cat, 'c_'+cat]
        f_name = os.path.join(destPath,"svm_"+cat+".pkl")
        with open(f_name, "rb") as fid:
            svm_model = cPickle.load(fid)
        y_prob[cat] = svm_model.predict_proba(X)[:,1]
        #print "Predicted for class {}" .format(cat)
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
        
    return pred
        

def generate_submission_file(pred, feature ,est):
    print "Generate submission file !"
    out_dict = {'version':version}
    ext_data_dict = {'used': True, 'details': \
                'ImageNetShuffle features.'}
    
    out_dict['results'] = pred
    out_dict['external_data'] = ext_data_dict        
    json_filename = 'sub_t3_'+feature+'_'+str(est)+'_'+SUBSET+'.json'
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
    
    # Temporal Proposals 
    # Optimized for high recall
    # 19994 x Mi proposals (For each video a number of proposals 
    # each with a score in decreasing order)
    
    # MBH
    # 19994 x 65536  features
    
#    X_train, X_val, nFeat = partition_dataset("MBH", train_vids_all, val_existing_vids)
#    nEst_mbh = [300]
#
#    for est in nEst_mbh:
#        print "MBH: For estimator value : %d" %est
#        #train_on_glFeat(X_train,nFeat, database, category_names, train_vids_all, "mbh_RF", 567, est)
#        print "MBH: Predictions for : %d" %est
#        pred_mbh, y_prob_mbh = predict_on_glFeat(X_val, nFeat, database, \
#                                             category_names, val_existing_vids, \
#                                             "mbh_RF", est)
        #generate_submission_file(pred_mbh, "MBH", est)
        
    
    # ImageNet Shuffle Features
    # 19994 x 1024 features
    nEst_sh = [200]
    X_train, X_val, nFeat = partition_dataset("SHUFFLE", train_vids_all, val_existing_vids)
    for est in nEst_sh:
        print "SH: For estimator value : %d" %est
        #train_on_glFeat(X_train, nFeat, database, category_names, train_vids_all, "sh_RF", 123, est)
        print "SH: Predictions for : %d" %est
        pred_sh, y_prob_sh = predict_on_glFeat(X_val, nFeat, database, \
                                            category_names, val_existing_vids, \
                                            "sh_RF", est)
        #generate_submission_file(pred_sh, "SH", est)
        
        
    # C3D Feature Predictions
    import caffe
    import frame_prediction as fp
    caffe.set_mode_gpu()
    net = caffe.Net(MODEL, PRETRAINED, caffe.TEST)

    # Predict on the validation set videos for each value of bgThreshold
    pred, y_prob_c3d = fp.get_predictions(net, val_existing_vids, category_names)
    
    #print y_prob_c3d.head()    
    
    
    # Combine scores across the models of same class
    ###########################################################################
    #X_temp = pd.DataFrame(np.random.random((len(X_val), len(category_names))), \
    #             columns=category_names, index=X_val.index)
    # only one of the two lines below should be uncommented.
    # first train the models and then predict for the test set
    print "Train and Predict using level 2 classifiers ... "
    train_level2_SVM( y_prob_sh, y_prob_c3d, \
                     X_val.loc[:,category_names], "lvl2_svm")
    pred = predict_on_SVM( y_prob_sh, y_prob_c3d,  "lvl2_svm")
    generate_submission_file(pred, "SVM", 0)
    ###########################################################################
    # Consider Taxonomy of the classes
    # Temporal Proposals
    # C3D features
    
    ###########################################################################


    


#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 17:21:26 2017

@author: Arpan

Description: ActivityNet -- Training
"""
import json
import os
import utils
import collections
import training_model_svm as tm1


# Server Params
#VIDEOPATH = '/home/arpan/DATA_Drive/ActivityNet/videos'
#JSONFILE = '/home/arpan/DATA_Drive/ActivityNet/ActivityNet-master/Evaluation/data/activity_net.v1-3.min.json'
#LMDB_FOLDER = "/home/arpan/DATA_Drive/ActivityNet"

# Local Params
VIDEOPATH = '/home/hadoop/VisionWorkspace/ActivityNet/ActivityNet-master/Crawler/videos'
JSONFILE = '/home/hadoop/VisionWorkspace/ActivityNet/ActivityNet-master/Evaluation/data/activity_net.v1-3.min.json'
LMDB_FOLDER = "/home/hadoop/VisionWorkspace/ActivityNet/new_lmdb/new2_lmdb"
SUBSET = 'training'
###############################################################################

# Train on HOG descriptors
# Iterate over the catogories and for each category train an SVM model
def sample_activity_frames(database, meta_info, category_names, N):
    """ Function samples N frame positions from the annotated video segments
    of each activity category. For 200 categories, 200 files will be created.
    Each file will have a 
    Input:dataframe cell value using column name and row no
    database: dictionary from activity_net.v1-3.min.json
    meta_info: dictionary of meta_information for training videos
        {'3aQnQEL3USQ':{u'total_frames': 6238, 
                        u'dimensions': [360, 480], u'fps': 29.5} ....}
    category_names: list of category_names, (sorted)
    Output: Write json files of the form
    {"vMYPNyBR3d0": [327, 327, 337, 345, 346, 359],...}
    Each file has N positions of activities, key is video-id and positions sampled 
    from that video
    """
    print "Called train_m1 !!!"
    video_ids = meta_info.keys()
    #N = 5000    # No of samples of each class to be picked from activity seq
    dest_folder = "samples_"+str(N)
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    # get list of training videos which belong to category
    for cat in category_names:
        print "Iterate for category %s" %cat
        video_ids_for_cat = utils.get_videos_for_category(database, \
                                                          video_ids,\
                                                          cat)
        # Retrieve samples from positive example videos.
        # Get dict of videos_ids and segments of videos which have action 
        # corresponding to a category.
        #print "Getting video segment information for %s videos..." %cat
        # train_segments dict: 
        train_segments = tm1.get_training_segments(database, video_ids_for_cat, cat)
        #print "Getting random frames from +ve example videos..."
        #print train_segments
        tr_samples = tm1.get_sample_frames(train_segments, meta_info, N)
        with open(os.path.join(dest_folder, cat+".json"), "w") as fp:
            json.dump(tr_samples, fp)
        
        # select 640x480 resolution and resize accordingly
        #break
        # retrieve samples from negative example videos


###############################################################################

def create_training_lmdb(srcSamplesMetaFiles, category_names):
    """ Loop over all the existing training videos
    category_names are sorted list of categories, where its index
    represents the category no.
    Path for json files of category videos and sample frame info: samples_5000
    This function assumes that you have already called train_m1 and the 
    json files for each category are present in the path specified
    Steps: 
    1. Extract the optical flow visualizations from the training set for each
    category.
    2. Convert into lmdb database 
    3. Train a CNN on the lmdb database
    4. Save the trained model to disk
    Input: meta_info: same as in the function above
    """
    
    samples_files = [s+".json" for s in category_names]
    
    assert len(samples_files)==len(category_names)
    # check order of categories names matches with samples_files
    for idx,f in enumerate(samples_files):
        if not (category_names[idx] in f):
            print f
            print samples_files
            raise IOError("Order of categories does not match order of sample files.")
    
    import optical_flow as of
    of.construct_datasets(VIDEOPATH, LMDB_FOLDER, srcSamplesMetaFiles, \
                          samples_files, category_names)
    
    return


###############################################################################

def train_m3(database, train_video_ids, category_names):
    # Loop over all the existing training videos
    # category_names are sorted list of categories, where its index
    # represents the category no.
    
    for idx in train_video_ids:
        # for each video call a method to train an SVM
        tm1.train_svm(os.path.join(VIDEOPATH, "v_"+idx+".mp4"), \
                     database[idx]['annotations'], 10,  category_names)
        # break used to execute for only one video
        break
    
    return

###############################################################################

if __name__=='__main__':
    # Read the database, version and taxonomy from JSON file
    with open("data/activity_net.v1-3.min.json", "r") as fobj:
        data = json.load(fobj)

    database = data["database"]
    taxonomy = data["taxonomy"]
    version = data["version"]
    
    non_existing_videos = utils.crosscheck_videos(VIDEOPATH, JSONFILE)

    print "No of non-existing videos: %d" % len(non_existing_videos)
    
    train_vids_all = []
    [train_vids_all.append(x) for x in database if database[x]['subset']==SUBSET]
    
    # Find list of available training videos
    train_existing_vids = list(set(train_vids_all) - set(non_existing_videos))
    
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
    # We use the meta-information such as FPS, totalFrames and dimensions 
    # in order to obtain a lower and upper bound for the frame sampling
    # To write meta-information to a json file. Uncomment following 3 lines 
    # to generate the json file.
    meta_info = tm1.get_meta_info(VIDEOPATH, train_existing_vids)
    with open("training_data_meta_info.json", "w") as fp:
        json.dump(meta_info, fp)
    
    # Read the training videos meta_information from file. 
    #with open("val_data_meta_info.json", "r") as fobj:
    #    meta_info = json.load(fobj)
    
    ###########################################################################
    
    # Train models 
    
    n = 4000    # no of samples to extract for each category of training videos
    #sample_activity_frames(database, meta_info, category_names, N=n)
    
    # Method 1: Train a series of SVMs on the training set videos
    
    # Uncomment below 3 lines for viewing frames selected
#    with open("samples_"+str(n)+"/Applying sunscreen.json") as fp:
#        samples_d = json.load(fp)
#    tm1.display_sample_frames(samples_d, VIDEOPATH)
    
    # Method 2: Train a CNN from scratch on the consecutive frame OF 
    # visualization images. 
    #create_training_lmdb("samples_"+str(n), category_names)
    print "LMDB Created !!"
    
    # Method 3: Use existing or third-party pre-trained models
    # Features: C3D , MBH (Improved Dense Traj) , ImageNetShuffle
    
    # Extract 
    
    # FineTune models
    
    # Save the models to files
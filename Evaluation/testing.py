#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 17:23:43 2017

@author: Arpan

Description: ActivityNet -- Testing and submission file generation
"""

import collections
import commands
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import get_video_number_of_frames
from skimage.transform import resize
import cv2
import random

# Server Params
# VIDEOPATH = '/home/arpan/DATA_Drive/ActivityNet/videos'
VIDEO_PATH = "/home/hadoop/VisionWorkspace/ActivityNet/ActivityNet-master/Crawler/videos"
SUBSET = 'validation'


###########################################################################

def get_sample_frame_from_video(videoid, duration, start_time, end_time, \
                                video_path=VIDEO_PATH):
    filename = glob.glob(os.path.join(video_path, "v_%s*" % videoid))[0]
    nr_frames = get_video_number_of_frames(filename)
    fps = (nr_frames*1.0)/duration
    start_frame, end_frame = int(start_time*fps), int(end_time*fps)
    frame_idx = random.choice(range(start_frame, end_frame))
    cap = cv2.VideoCapture(filename)
    keepdoing, cnt = True, 1
    while keepdoing:
        ret, img = cap.read()
        if cnt==frame_idx:
            break
        assert ret==True, "Ended video and frame not selected."
        cnt+=1
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

###########################################################################

def get_random_video_from_activity(database, activity, subset="validation"):
    videos = []
    for x in database:
        if database[x]["subset"] != subset: continue
        xx = random.choice(database[x]["annotations"])
        if xx["label"]==activity:
            yy = {"videoid": x, "duration": database[x]["duration"],
                  "start_time": xx["segment"][0], "end_time": xx["segment"][1]}
            videos.append(yy)
    return random.choice(videos)

###########################################################################

def get_video_prediction(vid, category_names, model):
    # Read the video frames and predict categories with scores
    predictions_lst = []
    no_of_preds = np.random.randint(1,4)
    for i in range(no_of_preds):
        score = float(np.random.rand(1))
        label_idx = np.random.randint(200)
        label = category_names[label_idx]
        pred_dict = {'score': score, 'label':label}
        predictions_lst.append(pred_dict)
    return predictions_lst


if __name__=='__main__':
    with open("data/activity_net.v1-3.min.json", "r") as fobj:
        data = json.load(fobj)

    database = data["database"]
    taxonomy = data["taxonomy"]
    version = data["version"]

    ###########################################################################
    # Release Summary
    all_node_ids = [x["nodeId"] for x in taxonomy]
    print len(all_node_ids)
    leaf_node_ids = []
    for x in all_node_ids:
        is_parent = False
        # iterate through the parentIds and if the nodeID is a parentId then
        # it is not a leaf node else it is a leaf node
        for query_node in taxonomy:
            if query_node["parentId"]==x: 
                is_parent = True
                break
        if not is_parent: leaf_node_ids.append(x)
        
    leaf_nodes = [x for x in taxonomy if x["nodeId"] in  leaf_node_ids]
    
    vsize = commands.getoutput("du %s -lhs" % VIDEO_PATH).split("/")[0]
    
    total_duration = sum([database[x]['duration'] for x in database])/3600.0

    print "ActivityNet %s" % version
    print "Total number of videos: %d" % len(database)
    print "Total number of nodes in taxonomy: %d" % len(taxonomy)
    print "Total number of leaf nodes: %d" % len(leaf_nodes)
    print "Total size of downloaded videos: %s" % vsize
    print "Total hours of video: %0.1f" % total_duration

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
    # Iterate over the validation/test set video files and obtain 
    # the prediction for each file
    subset_video_ids = []
    ext_data_dict = {'used': False, 'details': \
                'Describe the external data over here. If necessary for each prediction'}
    
    out_dict = {'version':version}
    
    [subset_video_ids.append(x) for x in database if database[x]['subset']==SUBSET]
    results_dict = {}
    for v_id in subset_video_ids:
        results_dict[v_id] = get_video_prediction(v_id, category_names, "")
    
    out_dict['results'] = results_dict
    out_dict['external_data'] = ext_data_dict
            
    json_filename = 'submission_'+SUBSET+'.json'
    with open(json_filename, 'w') as fp:
        json.dump(out_dict, fp)
    
    
    # write the out_dict to a JSON file
    ###########################################################################

#    plt.figure(num=None, figsize=(18, 8), dpi=100)
#    xx = np.array(category_count.keys())
#    yy = np.array([category_count[x] for x in category_count])
#    xx_idx = yy.argsort()[::-1]
#    plt.bar(range(len(xx)), yy[xx_idx], color=(240.0/255.0,28/255.0,1/255.0))
#    plt.ylabel("Number of videos per activity ")
#    plt.xticks(range(len(xx)), xx[xx_idx], rotation="vertical", size="small")
#    plt.title("ActivityNet VERSION 1.2 - Untrimmed Video Classification")
#    plt.show()

    ###########################################################################
    
    # read a model 
    
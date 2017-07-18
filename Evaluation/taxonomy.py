#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 04:24:44 2017
@author: Arpan
Description: Taxonomy generation

"""
import json
import numpy as np
import utils
import collections


VIDEOPATH = '/home/hadoop/VisionWorkspace/ActivityNet/ActivityNet-master/Crawler/videos'
JSONFILE = '/home/hadoop/VisionWorkspace/ActivityNet/ActivityNet-master/Evaluation/data/activity_net.v1-3.min.json'
SUBSET = 'validation'

def get_parentnode(taxonomy, nodeName):
    """
        Retrieve the parentId of a node given its label
    """    
    for entry in taxonomy:
        if entry['nodeName'] == nodeName:
            return entry['parentId']
    print "Node Name {} is invalid !" .format(nodeName)
    return -1

#def trace_path_to_root(taxonomy, label):
    
def get_nodeName(taxonomy, nodeId):
    """
        Retrieve nodeId given a nodeName
    """
    
def get_nodeId(taxonomy, nodeName):
    """
        Retrieve nodeID from the given nodeName
    """
    for entry in taxonomy:
        if nodeName == entry['nodeName']:
            return entry['nodeId']
    print "Node Name {} is invalid !" .format(nodeName)
    return -1

def nAIntersectB(database, taxonomy, train_vids_all):
    n = 0
    
def findDiscripancies(taxonomy):
    """
    Found nodeId 269 and 270 have same names 'Health-related self care'
    """
    i = 0
    for entry in taxonomy:
        if entry['parentName'] != None:
            print entry['nodeName']
            if entry['nodeName'].lower() == entry['parentName'].lower():
                i += 1
    print "No of same nodes = {} " .format(i)

def get_no_of_annotations(database, label, train_vids_all):
    """
    Iterate over the training videos and count the no of egs belonging to class i
    """
    count = 0
    for vid in train_vids_all:
        for ann in database[vid]['annotations']:
            if ann['label'] == label:
                count += 1
    return count
    


def display_all_paths(taxonomy):
    """
        Iterate over all the entries of the taxonomy dict and for each display the 
        path from that node to the root node.
    """
    for i,entry in enumerate(taxonomy):
        print "For nodeId : {} :: NodeName : {} " .format(entry['nodeId'], entry['nodeName'])
        parentId = entry['parentId']
        parentName = entry['parentName']
        while parentId != None:
            print "ParentId : {} :: ParentName : {}" .format(parentId, parentName)
            # Search for nodeId == parentId
            for temp in taxonomy:
                if temp['nodeId'] == parentId:
                    parentId = temp['parentId']
                    parentName = temp['parentName']
                    break
        if i == 5:
            break
        

if __name__ == '__main__':
    
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

    display_all_paths(taxonomy)
    #findDiscripancies(taxonomy)
    
    for cat in category_names:
        ncat = get_no_of_annotations(database, cat, train_vids_all)
        print "category {} :: |vids| {}" .format(cat, ncat)
    
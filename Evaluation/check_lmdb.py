#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 22:20:12 2017

@author: hadoop
Description: Read LMDB

"""
import lmdb
import numpy as np
import caffe
import cv2

#LMDB_PATH = "/home/hadoop/VisionWorkspace/KTH_OpticalFlow/dataset/kth_actions_train/LMDB/OF_lmdb"
LMDB_PATH = '/home/hadoop/VisionWorkspace/ActivityNet/new_lmdb/train_hog_lmdb'

def waitTillEscPressed():
    while(True):
        if cv2.waitKey(10)==27:
            print("Esc Pressed")
            return

if __name__ == '__main__':
    
    env = lmdb.open(LMDB_PATH, readonly=True)
    print env.stat()
    i,j = 0, 0
    with env.begin() as txn:
        cursor = txn.cursor()
        datum = caffe.proto.caffe_pb2.Datum()
        for k,v in cursor:
            datum.ParseFromString(v)
            lab = datum.label
            #print "Shape : {}" .format(datum.width)
            #flat_x = np.fromstring(datum.data, dtype=np.uint8)
            flat_x = np.array(datum.float_data)
            x = flat_x.reshape(datum.channels, datum.height, datum.width)
            y = datum.label
            #print "sum(x) = {} " .format(np.sum(x))
            #print "y = %d " %y
            j += 1
            if np.sum(x) == 0:
                print j
                print "class %d " %y
                i += 1
        #raw_datum = txn.get(b'00000000')
        
    print 'No of 0s are %d ' %i
#


#label = datum.label
#    data = caffe.io.datum_to_array(datum)
#    for l, d in zip(label, data):
#            print l, d

# Iterate over the LMDB values

#with env.begin() as txn:
#    cursor = txn.cursor()
#    datum = caffe.proto.caffe_pb2.Datum()
#    for key, value in cursor:
#        datum.ParseFromString(value)
#        label = datum.label
#        flat_x = np.fromstring(datum.data, dtype=np.uint8)
#        x = flat_x.reshape(datum.channels, datum.height, datum.width)
#        img = convert_to_bgr(x)
#        cv2.imshow("BGR_OF", img)
#        print "Label = "+str(label)
#        keyPressed = waitTillEscPressed()
#        if keyPressed==0:      # write to file
#            cv2.imwrite(os.path.join(curr_path,key+"_"+str(label)+".jpg"),img)
        #if key == '00000099':
        #    print(key, value)
    
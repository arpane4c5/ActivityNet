# My attempt at the ActivityNet Challenge 2017
This is my attempt at the [ActivityNet Challenge 2017](http://activity-net.org/challenges/2017/index.html). Thanks to the organizers for providing the boilerplate code and annotated datasets. [Link](https://github.com/activitynet/ActivityNet) to repository.

## Overview of the challenge
The ActivityNet Challenge 2017 had 5 tasks for activity recognition from videos. The datasets of ActivityNet (having 200 classes) and Kinetics (having 400 classes) were used. For details about the tasks and metrics used, do visit their [website](http://activity-net.org/challenges/2017/index.html)

## For executing the code.
* Define the paths in training.py for the meta-files and LMDBs to be saved. The path for the set videos must be correctly defined and it should contain all .mp4 files.

* Define the parameters for the number of samples to be generated for training and validation. 

* training.py file creates the LMDB and meta-files and saves them to disk.

* The folder caffe-models contains the details of the FC-Networks and other models that were tried out.

* Different files named as training\_model

### Major Requirements
1. Python 2.7 

2. Caffe (with Python wrappers)

3. OpenCV 3.2.0 (any version >2.4.0 will work. For 2.X version you may need to edit a few lines)

4. GPU card + CUDA Tools
 

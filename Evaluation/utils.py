import json
import urllib2
import cv2

import numpy as np

API = 'http://ec2-52-11-11-89.us-west-2.compute.amazonaws.com/challenge17/api.py'

def get_blocked_videos(api=API):
    api_url = '{}?action=get_blocked'.format(api)
    req = urllib2.Request(api_url)
    response = urllib2.urlopen(req)
    return json.loads(response.read())

def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap

def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.

    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.

    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
      + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU

def wrapper_segment_iou(target_segments, candidate_segments):
    """Compute intersection over union btw segments
    Parameters
    ----------
    target_segments : ndarray
        2-dim array in format [m x 2:=[init, end]]
    candidate_segments : ndarray
        2-dim array in format [n x 2:=[init, end]]
    Outputs
    -------
    tiou : ndarray
        2-dim array [n x m] with IOU ratio.
    Note: It assumes that candidate-segments are more scarce that target-segments
    """
    if candidate_segments.ndim != 2 or target_segments.ndim != 2:
        raise ValueError('Dimension of arguments is incorrect')

    n, m = candidate_segments.shape[0], target_segments.shape[0]
    tiou = np.empty((n, m))
    for i in xrange(m):
        tiou[:, i] = segment_iou(target_segments[i,:], candidate_segments)

    return tiou

def get_video_number_of_frames(filename):
    cap = cv2.VideoCapture(filename, 0)
    if not cap.isOpened():
        print "Error reading the video file. Abort."
        exit(0)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return length

def get_videos_for_category(database, video_ids, category):
    """ 
    Find the list of video IDS which belong to a particular category. 
    Search among the given list of video_ids as key values
    Note that only the training set existing_video_ids are to be used here.
    """
    video_ids_for_cat = []
    
    for idx in video_ids:
        ann_lst = database[idx]['annotations']
        for label in ann_lst:       # Iterate over each dictionary of label
            if label['label']==category:
                video_ids_for_cat.append(idx)
                break
    return video_ids_for_cat


def crosscheck_videos(video_path, ann_file):
    """ Get the list of videos present in the video_path and find the non-existing
    videos after reading all the videoIDs from the database json file.
    Return the list of non-existing videos
    """
    # Get existing videos
    import glob
    import os
    existing_vids = glob.glob("%s/*.mp4" % video_path)
    for idx, vid in enumerate(existing_vids):
        basename = os.path.basename(vid).split(".mp4")[0]
        if len(basename) == 13:
            existing_vids[idx] = basename[2:]
        elif len(basename) == 11:
            existing_vids[idx] = basename
        else:
            raise RuntimeError("Unknown filename format: %s", vid)
    # Read an get video IDs from annotation file
    with open(ann_file, "r") as fobj:
        anet_v_1_0 = json.load(fobj)
    all_vids = anet_v_1_0["database"].keys()
    non_existing_videos = []
    for vid in all_vids:
        if vid in existing_vids:
            continue
        else:
            non_existing_videos.append(vid)
    return non_existing_videos
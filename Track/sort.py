import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
from filterpy.kalman import KalmanFilter

from utils import IOU, convert_box, convert_to_bbox



class KalmanTracker(object):
    """
        Kalman Tracker for object tracking implemented from the paper 
        Tracking Multiple Moving Objects Using Unscented Kalman Filtering Techniques
        Xi Chen,Xiao WangandJianhua Xuan
    """
    count = 0
    def __init__(self, bbox):
         """
        Initialises a tracker using initial bounding box.
        """
        #define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z = 4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.objclass = bbox[6]

    def update(self, bbox):
        """
            Updates the state vector
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
    
    def predict(self):
        """
            Advances the state vector and returns estimate
        """
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
            self.kf.predict()
            self.age += 1
        
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        
        return self.history[-1]
    
    def get_state(self):
        return convert_to_bbox(self.kf.x)


def assosciate_detections(detections, trackers, iou_thresh = 0.4):
    """
        Assosciates trackers to detections.

    """
    if len(trackers) == 0:
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
    
    iou_matrix = np.zeros((len(detections), len(trackers)))

    # assignment cost matrix
    for d, dtc in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = IOU(dtc, trk)

    matched_indices = linear_assignment(-iou_matrix)

    unmatched_detections = [], unmatched_trackers = []
    for d,_ in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    for t,_ in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)
    
    matched = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_thresh:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matched.append(m.reshape(1, 2))

    if len(matched) == 0:
        matched = np.empty((0, 2), dtype=int)

    return matched, np.array(unmatched_detections), np.array(unmatched_trackers)
    
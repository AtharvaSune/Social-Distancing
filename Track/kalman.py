from sklearn.utils.linear_assignment_ import linear_assignment
from filterpy.kalman import KalmanFilter
import numpy as np
from .track_utils import *

class KalmanBoxTracker(object):
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
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])

        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_box(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.objclass = bbox[-1]

    def update(self, bbox):
        """
            Updates the state vector
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_box(bbox))

    def predict(self):
        """
            Advances the state vector and returns estimate
        """
        if((self.kf.x[6]+self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
            self.kf.predict()
            self.age += 1

        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return convert_to_bbox(self.kf.x[:4])

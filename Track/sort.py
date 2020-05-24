import numpy as np
from utils import IOU, convert_box, convert_to_bbox, assosciate_detections
from kalman import KalmanBoxTracker

class SORT(object):
    """
        Class to implement SORT Algorithm
    """
    def __init__(self, max_age=1, min_hits=3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 1

    def update(self, detections):
        """
            params:
                detections: a numpy array of detections
            this method is called once for each frame even if the detections
            is an empty numpy array

            return:
                array similar to detections
        """
        self.frame_count += 1

        # first update current trackers
        trks = np.zeros(shape=(len(self.trackers), 5))
        to_del = [] # list of trackers that need to be deleted.
        ret = [] # list to be returned
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            # if update for any of the tracker gives nan then that tracker is added
            # to list of trackers that need to be deleted.
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_detections, unmatched_trackers = assosciate_detections(detections, trks)

        # update trackers with matched detections
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trackers:
                d = matched[np.where(matched[:,1]==t)[0], 0]
                trk.update(detections[d, :][0])

        # create trackers for unmatched detections
        for i in unmatched_detections:
            trk = KalmanBoxTracker(detections[i, :])
            self.trackers.append(trk)

        i = len(self.trackers)
        # For all trackers
        for track in reversed(self.trackers):
            # get back the bounding box
            d = track.get_state()[0]

            # if they satisfy the required criteria add them to return list
            if((track.time_since_update < 1) and (track.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
                ret.append(np.concatenate((d, [track.id+1], [track.objclass])).reshape(1, -1))
            i -= 1

            # else delete tracker
            if(track.time_since_update > self.max_age):
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))
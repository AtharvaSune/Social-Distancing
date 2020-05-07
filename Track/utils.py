import numpy as np

def IOU(bb_test, bb_gt):
    """
        Compute IOU of two boxes 
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    
    # overlap dimensions
    w = np.maximum(0, xx2 - xx1)
    h = np.maximum(0, yy2 - yy1)

    # IOU = overlap_area / union_area
    iou = w*h/ ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
    + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - w*h) 

    return iou

def convert_box(bbox):
    """
        Takes bbox in the format [x1, y1, x2, y2]:
        x1, y1 = top left
        x2, y2 = bottom right
        
        returns in format [x, y, s, r]:
        x = horizontal coordinate of center
        y = vertical coordinate of center
        s = scale
        r = aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2
    y = bbox[1] + h/2
    s = w*h
    r = w/float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_to_bbox(cvted):
    """
        Takes converted bbox in the form [x, y, s, r] and returns in 
        format [x1, y1, x2, y2]
    """
    (x, y, s, r) = cvted
    w = np.sqrt(s*r)
    h = s/float(w)
    x1 = x-w/2
    y1 = y+w/2
    x2 = x+w/2
    y2 = y-w/2
    return np.array([x1, y1, x2, y2]).reshape((4,1))
    

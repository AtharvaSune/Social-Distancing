from Track.sort import SORT
from Detect.frcnn import get_detections
import torch
import numpy as np
import cv2
from PIL import Image


def socialDistancing(vid_path, out_path):
    cap = cv2.VideoCapture(vid_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255),
              (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 0, 128),
              (128, 128, 0), (0, 128, 128)]

    writer = cv2.VideoWriter(out_path + "social_distance.mp4",
                             cv2.VideoWriter_fourcc(*"MJPG"),
                             10, size)

    tracker = SORT()
    while True:
        grabbed, frame = cap.read()
        if not grabbed:
            break

        frame = cv2.resize(frame, (1024, 1024))
        preds = get_detections(Image.fromarray(frame))[0]
        boxes = preds["boxes"]
        scores = preds["scores"].view(boxes.size(0), 1)
        labels = preds["labels"].view(boxes.size(0), 1)
        detection = None
        for box, score, label in zip(boxes, scores, labels):
            if score.item() >= 0.9:
                d = torch.cat([box, score.float(), label.float()], dim=-1).view(1, -1)
                if detection is None:
                    detection = d
                else:
                    detection = torch.cat([detection, d], dim=0)

        tracked_objects = tracker.update(detection)
        unique_labels = detection[:, -1].unique()
        n_cls_preds = len(unique_labels)
        red = {}

        for i in range(len(tracked_objects)):
            red[i] = []

        for i in range(len(tracked_objects)):
            x11, y11, x12, y12, id1, _ = tracked_objects[i]
            cent_1 = np.array([(x11 + x12)/(2*frame_width),
                               (y11 + y12)/(2*frame_height)])

            for j in range(i+1, len(tracked_objects)):
                x21, y21, x22, y22, id2, _ = tracked_objects[j]
                cent_2 = np.array([(x21 + x22)/(2*frame_width),
                                   (y21 + y22)/(2*frame_height)])
                if np.sqrt((np.sum((cent_1 - cent_2)**2))) < 0.8:
                    red[i].append(j)

        colored = []
        print(red)
        for idx in red.keys():
            x1, y1, x2, y2, obj_id, _ = tracked_objects[idx]
            w = int(x2-x1)
            h = int(y2-y1)
            x1 = int(x1)
            y1 = int(y1)
            cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (0, 0, 255), 3)
            cv2.putText(frame, f"{obj_id} violate", (x1, y1+10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, 
                        (255, 255, 255), 3)
            colored.append(idx)
            for idx_close in red[idx]:
                x1, y2, x2, y2, obj_id, _ = tracked_objects[idx_close]
                w = int(x2-x1)
                h = int(y2-y1)
                x1 = int(x1)
                y1 = int(y1)
                cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (0, 0, 255), 3)
                cv2.putText(frame, f"{obj_id} violate", (x1, y1+10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 3)
                colored.append(idx_close)

        colors = [(128, 0, 128), (122, 122, 0), (255, 128, 0)]

        for i, (x1, y1, x2, y2, obj_id, _) in enumerate(tracked_objects):
            w = int(x2-x1)
            h = int(y2-y1)
            x1 = int(x1)
            y1 = int(y1)
            cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (255, 255, 255), 3, 3)
            cv2.putText(frame, str(obj_id), (x1, y1+10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 3)

        writer.write(cv2.resize(frame, size))
import torch
import argparse
import numpy as np
import cv2
from PIL import Image
from Detect.frcnn import get_detections
from Track.sort import SORT


def track(vid_path, out_path):
    cap = cv2.VideoCapture(vid_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255),
              (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 0, 128),
              (128, 128, 0), (0, 128, 128)]

    writer = cv2.VideoWriter(out_path + "track.mp4",
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

        for x1, y1, x2, y2, id, class_pred in tracked_objects:
            box_h = int((y2-y1))
            box_w = int((x2-x1))
            x1 = int(x1)
            y1 = int(y1)
            color = colors[int(id) % len(colors)]
            cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
            cv2.putText(frame, str(id), (x1+10, y1+10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 3)

        writer.write(cv2.resize(frame, size))

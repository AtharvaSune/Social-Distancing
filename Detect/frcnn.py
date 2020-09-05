import torch
import argparse
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import transforms as T
import numpy as np
import cv2
from PIL import Image

model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
if torch.cuda.is_available():
    model.cuda()


def get_detections(frame):
    transform = T.Compose([
        T.Resize(1024),
        T.ToTensor()
    ])
    frame = transform(frame)
    if torch.cuda.is_available():
        frame = frame.to("cuda:0")
    with torch.no_grad():
        predictions = model([frame])
    for i in predictions:
        for x in i:
            i[x] = i[x].cpu()
    return predictions


def image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (1024, 1024))
    preds = get_detections(Image.fromarray(img))[0]
    boxes = preds["boxes"]
    scores = preds["scores"]
    labels = [int(i) for i in preds["labels"]]

    colors = {x: np.random.randint(0,256, (3,)) for x in set(labels)}
    for i, box in enumerate(boxes):
        label = labels[i]
        if label != 1:
            continue
        color = [int(i) for i in colors[label]]
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color , 2)

    cv2.imwrite("test2.jpg", cv2.resize(img, (1000, 667)))


def video(vid_path, out_path):
    cap = cv2.VideoCapture(vid_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)

    writer = cv2.VideoWriter(out_path + "object.mp4", cv2.VideoWriter_fourcc(*"MJPG"), 10, size)

    while True:
        grabbed, img = cap.read()
        if not grabbed:
            break

        img = cv2.resize(img, (1024, 1024))
        preds = get_detections(Image.fromarray(img))[0]
        boxes = preds["boxes"]
        scores = preds["scores"]
        labels = [int(i) for i in preds["labels"]]

        colors = {x: np.random.randint(0, 256, (3,)) for x in set(labels)}
        for i, box in enumerate(boxes):
            if scores[i] <= 0.8:
                continue
            label = labels[i]
            color = [int(i) for i in colors[label]]
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color , 2)

        img = cv2.resize(img, size)
        writer.write(img)
        # cv2.imshow(args.path, cv2.resize(img, (1000, 667)))
        if cv2.waitKey(25) & 0xff == ord("q"):
            break

    writer.release()
    cap.release()

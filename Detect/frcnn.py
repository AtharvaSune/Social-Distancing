import torch
import argparse
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import transforms as T
import numpy as np

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Object detector")
    parser.add_argument("-i", "--image", help="If image", action="store_true")
    parser.add_argument("-v", "--video", help="If Video", action="store_true")
    parser.add_argument("-p", "--path", help="path to test image/video", required="true", type=str)

    args = parser.parse_args()

    if args.image:
        try:
            import cv2
            from PIL import Image

            img = cv2.imread(args.path)
            img = cv2.resize(img, (1024, 1024))
            preds = get_detections(Image.fromarray(img))[0]
            print(preds)
            boxes = preds["boxes"]
            scores = preds["scores"]
            labels = [int(i) for i in preds["labels"]]

            colors = {x: np.random.randint(0,256, (3,)) for x in set(labels)}
            for i, box in enumerate(boxes):
                # if scores[i] <= 0.5:
                #     continue
                label = labels[i]
                if label != 1:
                    continue
                color = [int(i) for i in colors[label]]
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=color ,thickness=2)
                
            cv2.imwrite("test2.jpg", cv2.resize(img, (1000, 667)))

        except EnvironmentError as e:
            print("Some error happened")

    elif args.video:
        try:
            import cv2
            from PIL import Image

            cap = cv2.VideoCapture(path)
            while True:
                grabbed, frame = cap.read()
                if not grabbed:
                    break

                img = cv2.resize(img, (1024, 1024))
                preds = get_detections(Image.fromarray(img))[0]
                boxes = preds["boxes"]
                scores = preds["scores"]
                labels = [int(i) for i in preds["labels"]]

                colors = {x: np.random.randint(0,256, (3,)) for x in set(labels)}
                for i, box in enumerate(boxes):
                    if scores[i] <= 0.5:
                        continue
                    label = labels[i]
                    color = [int(i) for i in colors[label]]
                    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=color ,thickness=2)

                cv2.imshow(args.path, cv2.resize(img, (1000, 667)))
                if cv2.waitKey(25) & 0xff == ord("q"):
                    break
        except:
            print("Error Loading Video")


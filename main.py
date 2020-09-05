import argparse

from Detect.frcnn import video
from object_tracking import track
from social_distancing import socialDistancing


def main(vid_path, out_path):
    # video(vid_path, out_path)
    # track(vid_path, out_path)
    socialDistancing(vid_path, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tracking vs Detection")
    parser.add_argument("--vid_path", required=True, type=str, help="path to video to track objects")
    parser.add_argument("--out_path", required=True, type=str, help="path to output video")

    args = parser.parse_args()

    main(args.vid_path, args.out_path)

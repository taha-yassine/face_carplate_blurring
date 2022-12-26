import argparse
import os
import platform
import sys
from pathlib import Path
import cv2

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


@torch.inference_mode()
def run(
        device='',  # cuda device, i.e. cuda (or gpu ordinal) or cpu
        source='', # file/dir
):
    # Path to weights
    path = 'best.pt'

    # Load file
    cap = cv2.VideoCapture(source)
    if not cap.isOpened(): 
        print("Error opening video stream or file.")
        exit()

    # Load model
    device = select_device(device)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=path, device=device)
    
    # Read until video is completed
    while True:
        ret, im = cap.read()

        if ret:
            res = model(im)
            for c in res.crop(save=False):
                x0,y0,x1,y1=[int(x.item()) for x in c['box']]
                # im = cv2.rectangle(im,(x0,y0),(x1,y1),(0, 0, 255),2) # Draw detection rectangle
                im[y0:y1,x0:x1] = cv2.blur(im[y0:y1,x0:x1], (23,23)) # Blur detection region

            cv2.imshow('Frame',im)

            # Press 'q' to exit
            if cv2.waitKey(10)&0xFF == ord('q'):
                break

        else:
            break
        
    cap.release()
    cv2.destroyAllWindows()


def select_device(device=''):
    if device.isdigit():
        return torch.device(int(device))
    elif device in ['cuda','cpu']:
        return torch.device(device)
    raise Exception('Selected device is invalid. Please select a valid device.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='', help='cuda device, i.e. cuda (or gpu ordinal) or cpu')
    parser.add_argument('--source', type=str, help='file/dir')
    opt = parser.parse_args()

    try:
        run(**vars(opt))
    except Exception as e:
        print(e)

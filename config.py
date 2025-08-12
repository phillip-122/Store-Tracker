import torch
import numpy as np
from pathlib import Path
import supervision as sv
from ultralytics import YOLO
from torchreid.utils import FeatureExtractor

baseDirectory = Path.cwd()

#change these to wherever you want to store the video
FILE_PATH = baseDirectory / 'Videos' / 'video_footage.mp4'
# FILE_PATH = baseDirectory / 'Videos' / 'video_footage_shortened.mp4'

TARGET_PATH = baseDirectory / 'output.mp4'

CSV_OUTPUT = baseDirectory / 'customer_log.csv'

#This is where I store my tesseract.exe, change to whatever yours is
TESSERACT_PATH = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")

#This is for what threshold you want to consider a person the same as another person
REID_THRESHOLD = 0.7

#This is the model that the reid uses, in my case I am using osnet, but you can change it to other models if you would like to try them out
extractor = FeatureExtractor(
    model_name='osnet_x0_25',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

#This is the yolo model I chose because it is very accurate but not super slow
# if you want it to run faster then switch the model to a lower version such as m
model = YOLO("yolo11l.pt")

LINE_TOP = sv.Point(x=200, y=0)
LINE_BOTTOM = sv.Point(x=200, y=900)

WORKER_ZONE = np.array([
    [0, 1079],
    [0, 850],
    [940, 950],
    [970, 1079]
])

GLASSES_ZONE = np.array([
    [660, 0],
    [1919, 0],
    [1919, 1050], 
    [1300, 1919],
    [970, 830],
    [660, 740]
])

LEGIT_ENTRY_ZONE = np.array([
    [220, 0],
    [1919, 0],
    [1919, 1079],
    [220, 1079]
])

#if you want to view the time annotation
# timeZone = np.array([
#     [1660, 1000],
#     [1880, 1000],
#     [1880, 1070],
#     [1660, 1070]
# ])

TIME_X, TIME_Y = 1660, 1000
TIME_W, TIME_H = 220, 70
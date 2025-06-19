""" General configurations for main.py """
import os

# Input video path
# INPUT_VIDEO_PATH = "./examples/videos/rally.mp4"
INPUT_VIDEO_DIR = "/mnt/c/data/plai/rallies/CVSPORTS_Padel/rallies"
INPUT_VIDEO_NAME = "20231015_AMSTERDAM_04.mp4"
INPUT_VIDEO_PATH = os.path.join(INPUT_VIDEO_DIR, INPUT_VIDEO_NAME)

# Inference video path
OUTPUT_VIDEO_PATH = f"results_{INPUT_VIDEO_NAME}.mp4"

# True to collect 2d projection data
COLLECT_DATA = True
# Collected data path
COLLECT_DATA_PATH = "data.csv"

# Maximum number of frames to be analysed
MAX_FRAMES = None

# Fixed court keypoints
FIXED_COURT_KEYPOINTS_SAVE_PATH = f"./cache/fixed_keypoints_detection_{INPUT_VIDEO_NAME}.json"
FIXED_COURT_KEYPOINTS_LOAD_PATH = FIXED_COURT_KEYPOINTS_SAVE_PATH if os.path.isfile(FIXED_COURT_KEYPOINTS_SAVE_PATH) else None

# Players tracker
PLAYERS_TRACKER_MODEL = "./model_weights/players_detection/yolov8m.pt"
PLAYERS_TRACKER_BATCH_SIZE = 8
PLAYERS_TRACKER_ANNOTATOR = "rectangle_bounding_box"
PLAYERS_TRACKER_SAVE_PATH = f"./cache/players_detections_{INPUT_VIDEO_NAME}.json"
PLAYERS_TRACKER_LOAD_PATH = PLAYERS_TRACKER_SAVE_PATH if os.path.isfile(PLAYERS_TRACKER_SAVE_PATH) else None

# Players keypoints tracker
PLAYERS_KEYPOINTS_TRACKER_MODEL = "./model_weights/players_keypoints_detection/best.pt"
PLAYERS_KEYPOINTS_TRACKER_TRAIN_IMAGE_SIZE = 1280
PLAYERS_KEYPOINTS_TRACKER_BATCH_SIZE = 8
PLAYERS_KEYPOINTS_TRACKER_SAVE_PATH = f"./cache/players_keypoints_detections_{INPUT_VIDEO_NAME}.json"
PLAYERS_KEYPOINTS_TRACKER_LOAD_PATH = PLAYERS_KEYPOINTS_TRACKER_SAVE_PATH if os.path.isfile(PLAYERS_KEYPOINTS_TRACKER_SAVE_PATH) else None

# Ball tracker
BALL_TRACKER_MODEL = "./model_weights/ball_detection/TrackNet_best.pt"
BALL_TRACKER_INPAINT_MODEL = "./model_weights/ball_detection/InpaintNet_best.pt"
BALL_TRACKER_BATCH_SIZE = 8
BALL_TRACKER_MEDIAN_MAX_SAMPLE_NUM = 400
BALL_TRACKER_SAVE_PATH = f"./cache/ball_detections_{INPUT_VIDEO_NAME}.json"
BALL_TRACKER_LOAD_PATH = BALL_TRACKER_SAVE_PATH if os.path.isfile(BALL_TRACKER_SAVE_PATH) else None

# Court keypoints tracker
KEYPOINTS_TRACKER_MODEL = "./model_weights/court_keypoints_detection/best.pt"
KEYPOINTS_TRACKER_BATCH_SIZE = 8
KEYPOINTS_TRACKER_MODEL_TYPE = "yolo"
KEYPOINTS_TRACKER_SAVE_PATH = f"./cache/keypoints_detections_{INPUT_VIDEO_NAME}.json"
KEYPOINTS_TRACKER_LOAD_PATH = KEYPOINTS_TRACKER_SAVE_PATH if os.path.isfile(KEYPOINTS_TRACKER_SAVE_PATH) else None

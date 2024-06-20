import cv2
import onnx
import onnxruntime
import numpy as np
from YOLOv7 import YOLOv7
from utils import draw_detections
import math
from moviepy import editor
import os
import boto3
import io
from dotenv import load_dotenv
import os
import dlib
import datetime
from tqdm import tqdm


# Load environment variables from the .env file
load_dotenv()

# Access the environment variables
ACCESS_KEY_ID = os.getenv('ACCESS_KEY_ID')
SECRET_ACCESS_KEY = os.getenv('SECRET_ACCESS_KEY')
# debug = os.getenv('DEBUG')

s3_client = boto3.client(
    's3',
    aws_access_key_id=ACCESS_KEY_ID,
    aws_secret_access_key=SECRET_ACCESS_KEY
)
# Initialize the S3 client
# s3_client = boto3.client('s3')
bucket_name = 'resq'  # replace with your actual bucket name

class ProgressPercentage(object):
    def __init__(self, filename):
        self._filename = filename
        self._size = float(s3_client.head_object(Bucket=bucket_name, Key=filename)['ContentLength'])
        self._seen_so_far = 0
        self._lock = threading.Lock()
        
    def __call__(self, bytes_amount):
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            tqdm.write(f"{self._filename}  {self._seen_so_far}/{self._size}  ({percentage:.2f}%)")

def download_model_from_s3(model_key, local_path):
    # Using tqdm to display the progress bar
    with tqdm(total=int(s3_client.head_object(Bucket=bucket_name, Key=model_key)['ContentLength']), unit='B', unit_scale=True, desc=model_key) as pbar:
        s3_client.download_file(bucket_name, model_key, local_path, Callback=lambda bytes_transferred: pbar.update(bytes_transferred))


b = "yolov7_384x640.onnx"
download_model_from_s3("road_camera_models/yolov7_384x640.onnx", b)
c = "midas.onnx"
download_model_from_s3("road_camera_models/midas.onnx", b)



def is_tailgating(box, depth_map, depth_threshold=50):
    # Extract box coordinates
    x1, y1, x2, y2 = box.astype(int)

    # Calculate the mean depth within the box
    mean_depth = np.mean(depth_map[y1:y2, x1:x2])

    # Check if the mean depth is below the threshold (indicating proximity)
    return mean_depth < depth_threshold



def draw_detections_midas(image, boxes, mask_alpha=0.3, x_percentage=0.0, y_percentage=1.3, w_percentage=0.4, h_percentage=0.2):
    mask_img = image.copy()
    det_img = image.copy()

    # img_height, img_width = image.shape[:2]

    # Draw bounding boxes on the image
    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)

        # Calculate new coordinates based on percentages
        new_x = x1 + int((x2 - x1) * x_percentage)
        new_y = y1 + int((y2 - y1) * y_percentage)
        new_width = int((x2 - x1) * w_percentage)
        new_height = int((y2 - y1) * h_percentage)

        # Draw rectangle
        cv2.rectangle(det_img, (new_x, new_y), (new_x + new_width, new_y + new_height), (0, 255, 0), 2)

        # Draw fill rectangle in mask image
        cv2.rectangle(mask_img, (new_x, new_y), (new_x + new_width, new_y + new_height), (0, 255, 0), -1)

    return cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)



def run_object_detection_and_depth_estimation(image, depth_threshold=50):
    yolov7_model_path = "yolov7_384x640.onnx"
    yolov7_detector = YOLOv7(yolov7_model_path, conf_thres=0.5, iou_thres=0.5)

    midas_model_path = "midas.onnx"
    midas_session = onnxruntime.InferenceSession(midas_model_path, providers=['CUDAExecutionProvider'])
    midas_input_height, midas_input_width = 256, 256
    midas_output_width, midas_output_height = 800, 600

    # Load the image
    # frame = cv2.imread(image_path)
    frame = image

    # Perform lane finding pipeline
    # frame_with_lanes, poly_vertices = lane_finding_pipeline(frame.copy())

    # Create a mask for the region of interest (ROI) using the polygon vertices
    roi_mask = np.zeros_like(frame, dtype=np.uint8)
    # if poly_vertices:
    #     cv2.fillPoly(roi_mask, [np.array(poly_vertices)], (255, 255, 255))  # Convert poly_vertices to numpy array before passing

    # Apply the mask to the frame
    frame_roi = cv2.bitwise_and(frame, roi_mask)

    # Continue with the rest of the code...


    boxes, scores, class_ids = yolov7_detector(frame_roi)

    car_truck_indices = [i for i, class_id in enumerate(class_ids) if class_id in [2, 7]]
    car_truck_boxes = [boxes[i] for i in car_truck_indices]
    car_truck_scores = [scores[i] for i in car_truck_indices]

    detection_img = draw_detections(frame, car_truck_boxes, car_truck_scores, class_ids=class_ids)

    input_frame = cv2.resize(frame, (midas_input_width, midas_input_height))
    input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
    input_frame = input_frame.astype(np.float32) / 255.0
    input_data = input_frame.transpose(2, 0, 1).reshape(1, 3, midas_input_height, midas_input_width)

    depth_output = midas_session.run(None, {'input': input_data})[0].squeeze()

    resized_depth = cv2.resize(
        cv2.applyColorMap(
            cv2.convertScaleAbs(depth_output, alpha=255.0 / np.max(depth_output)),
            cv2.COLORMAP_JET
        ),
        (midas_output_width, midas_output_height)
    )

    depth_with_boxes = draw_detections_midas(resized_depth, car_truck_boxes)

    for box in car_truck_boxes:
        if is_tailgating(box, depth_output, depth_threshold):
            cv2.putText(depth_with_boxes, "Tailgating Alert!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    resized_detection_img = cv2.resize(detection_img, (detection_img.shape[1], depth_with_boxes.shape[0]))
    combined_img = np.concatenate((resized_detection_img, depth_with_boxes), axis=1)

    # cv2.imshow('Combined Object Detection and Depth Estimation', combined_img)

    for box in car_truck_boxes:
        print("Box Coordinates (YOLOv7):", box)

    return combined_img
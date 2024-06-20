import cv2
import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import time
import pymongo
from pyfcm import FCMNotification
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from kafka import KafkaConsumer
import numpy as np
from signal import signal, SIGPIPE, SIG_DFL
from raspberry_no_lanes import run_object_detection_and_depth_estimation
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
folder_name = 'front_camera_frames'

def save_frame_to_s3(frame, frame_number):
    _, buffer = cv2.imencode('.png', frame)
    frame_key = f"{folder_name}/{frame_number}.png"
    s3_client.put_object(Bucket=bucket_name, Key=frame_key, Body=buffer.tobytes())

# Function to delete frame from S3
def delete_frame_from_s3(frame_number):
    frame_key = f"{folder_name}/{frame_number}.png"
    s3_client.delete_object(Bucket=bucket_name, Key=frame_key)


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

# Download the ONNX models from S3 and store the paths in variables
a = "car.h5"
download_model_from_s3("road_camera_models/car.h5", a)

# Constants
frame_interval = 0.1
IMG_SIZE = 224
email_id = 'priyam22rr@gmail.com'
last_emergency_response_time = 0

# Kafka Configuration
topic = 'video'
consumer = KafkaConsumer(
    topic,
    bootstrap_servers=['13.201.34.30:9092'],
    fetch_max_bytes=26214400,
    max_partition_fetch_bytes=26214400
)

# Seek to the end for real-time data
try:
    consumer.poll()
except Exception as e:
    print(f"Error seeking to end: {e}")

# MongoDB and Email Notification Functions
def emergency_response(email_id):
    client = pymongo.MongoClient("mongodb+srv://priyam:pqrs.123@cluster0.1uefwpt.mongodb.net/")
    db = client["car_crash"]
    collection = db["user_login"]
    
    def import_data_by_email(email_id):
        try:
            data_cursor = collection.find({"email_id": email_id})
            df = pd.DataFrame(list(data_cursor))

            if not df.empty:
                return df
            else:
                print("No data found for the provided email ID.")
                return None

        except Exception as e:
            print("Error occurred while importing data:", e)
            return None
        finally:
            client.close()

    email_id_to_search = email_id
    result_df = import_data_by_email(email_id_to_search)

    if result_df is not None:
        print("Data imported successfully for email:", email_id_to_search)
        
    for i in range(len(result_df['_id'])):
        fcm = result_df['fcm'][i]
        
    def notify_crash(fcm_token, crash_info):
        push_service = FCMNotification(api_key="AAAAucIfw-w:APA91bHy03w5pMy4AVf14qKy7M1Bw0JXMm4_A19r_KuY1viHVL3ky7wsqa34oaceDCTsQWaB5dGwa4gnDDqDnch9VvRjcl-fQw1YAY_WxNvhtigD5NGDftJEUSKJMp2ePWd3pQGS_UNm")

        message_title = "Crash Detected"
        message_body = "A crash was detected at location X."

        result = push_service.notify_single_device(
            registration_id=fcm_token, 
            message_title=message_title, 
            message_body=message_body, 
            data_message=crash_info
        )

    notify_crash(fcm_token=fcm, crash_info={
        'crash_time': '2021-07-11 14:30:00',
        'crash_location': 'Lat: 40.7128, Lon: 74.0060',
        'crash_severity': 'High',
        # ... any other data you want to send ...
    })

    print("Done")
    
    for i in range(len(result_df['r_email'])):
        receiver_email = result_df['r_email'][i]
        
    def send_email(sender_email, receiver_email):
        subject = 'crash detected'
        message = 'crash alert!'

        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject
        msg.attach(MIMEText(message, 'plain'))

        smtp_server = 'smtp.gmail.com'
        smtp_port = 587

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, 'mlonvvatlfnilogu')
            server.send_message(msg)

    send_email(sender_email=email_id, receiver_email=receiver_email)
    print("email sent")

def can_call_emergency_response():
    global last_emergency_response_time
    current_time = time.time()
    return (current_time - last_emergency_response_time) >= 300  # 5 minutes in seconds

def call_emergency_response():
    print("function called")
    global last_emergency_response_time
    emergency_response(email_id=email_id)
    last_emergency_response_time = time.time()
    print("=============-=-=-=======================-=-=-===================-=-=-===")

def load_crash_detection_model(model_path):
    custom_objects = {'KerasLayer': hub.KerasLayer}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    return model

def process_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = tf.image.resize(frame_rgb, size=[IMG_SIZE, IMG_SIZE])
    frame_tensor = tf.convert_to_tensor(frame_resized, dtype=tf.float32) / 255.0
    frame_batch = tf.expand_dims(frame_tensor, axis=0)
    return frame_batch

def predict_crash(model, frame_batch):
    prediction = model.predict(frame_batch)
    return prediction

def extract_frame_and_process(frame, model):
    frame_batch = process_frame(frame)
    prediction = predict_crash(model, frame_batch)
    print(prediction)
    crash_probability = prediction[0][1]
    
    crash_detected = False
    if crash_probability > 0.80:
        if can_call_emergency_response():
            call_emergency_response()
            crash_detected = True

    return crash_detected, crash_probability

# Load the crash detection model
crash_detection_model_path = 'car.h5'
crash_detection_model = load_crash_detection_model(crash_detection_model_path)
frame_count = 0
# Kafka Consumer Loop
for msg in consumer:
    nparr = np.fromstring(msg.value, np.uint8)
    flags = cv2.IMREAD_COLOR
    frame = cv2.imdecode(nparr, flags)
    
    # Process frame for crash detection
    crash_detected, crash_probability = extract_frame_and_process(frame, crash_detection_model)
    print(f"Crash Detected: {crash_detected}, Crash Probability: {crash_probability}")

    # gray_frame = frame
    # gray_frame_path = "gray_frame.jpg"
    # cv2.imwrite(gray_frame_path, gray_frame)

    processed_frame = run_object_detection_and_depth_estimation(frame)

    frame_count += 1
    save_frame_to_s3(processed_frame, frame_count)

        # Delete the frame from S3
    if frame_count > 10:
        delete_frame_from_s3(frame_count - 10)

    # os.remove(gray_frame_path)

    # cv2.imshow('Raspberry Pi Video', processed_frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cv2.destroyAllWindows()
signal(SIGPIPE, SIG_DFL)

import cv2
import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
frame_interval = 0.1
import time
last_emergency_response_time = 0

import pymongo
from pyfcm import FCMNotification
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
#---------------------------------------------------------------------------------------------------------------------
# will come from dashcam
# video_path = "crash_video.mp4"
email_id = 'priyam22rr@gmail.com'
# time = t
#---------------------------------------------------------------------------------------------------------------------
IMG_SIZE = 224

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
        fcm
        
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

    notify_crash(fcm_token= fcm ,crash_info = {
        'crash_time': '2021-07-11 14:30:00',
        'crash_location': 'Lat: 40.7128, Lon: 74.0060',
        'crash_severity': 'High',
        # ... any other data you want to send ...
    })

    print("Done")
    
    
#     email:

    for i in range(len(result_df['r_email'])):
        receiver_email = result_df['r_email'][i]
        
    def send_email(sender_email, receiver_email):
    #     sender_email = 'priyam22rr@gmail.com'
    #     receiver_email = 'shivamvijayvargiya03@gmailcom'
        subject = 'crash detected'
        message = 'crash alert!'

        # Create a multipart message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject

        # Add the message body
        msg.attach(MIMEText(message, 'plain'))

        # SMTP server configuration
        smtp_server = 'smtp.gmail.com'
        smtp_port = 587  # Replace with the appropriate port for your server

        # Establish a secure connection with the SMTP server
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            # Login to your email account
            server.login(sender_email, 'mlonvvatlfnilogu')
            # Send the email
            server.send_message(msg)

    send_email(sender_email = email_id, receiver_email = receiver_email)
    print("email sent")
        
# emergency_response(email_id = email_id)

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
    # Load the crash detection model with custom objects
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
    # Process the incoming frame
    frame_batch = process_frame(frame)

    # Predict crash using the processed frame
    prediction = predict_crash(model, frame_batch)
    print(prediction)
    crash_probability = prediction[0][1]
    
    # Decide if it's a crash scenario based on the model's prediction
    crash_detected = False
    if crash_probability > 0.80:
        if can_call_emergency_response():
            call_emergency_response()
            crash_detected = True

    # Return whether a crash was detected and the crash probability
    return crash_detected, crash_probability



crash_detection_model_path = 'car.h5'
crash_detection_model = load_crash_detection_model(crash_detection_model_path)

# extract_frames(video_path, frame_interval, crash_detection_model)





from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from io import BytesIO
import cv2
import numpy as np
import os
from datetime import datetime

app = FastAPI()

origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a folder to save the frames
save_folder = "saved_frames"
save_folder1 = "saved_frames1"
os.makedirs(save_folder, exist_ok=True)
os.makedirs(save_folder1, exist_ok=True)

# Counter for serial naming
frame_counter = 1

# Endpoint to process the received frame using an OpenCV model
# Adjusted FastAPI endpoint to use the new function



# TAILGATING
from rasberry1 import run_object_detection_and_depth_estimation




@app.post("/api/process_frame/")
async def process_frame_api(file: UploadFile = File(...)):
    global frame_counter

    # Read the uploaded frame
    contents = await file.read()

    # Decode the frame
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Call the adjusted function to process the frame and predict crash
        ###########################################################################################################################################
    crash_detected, crash_probability = extract_frame_and_process(frame, crash_detection_model)
    print(crash_detected, crash_probability)
        ###########################################################################################################################################


    gray_frame = frame

    gray_frame_path = "gray_frame.jpg"
    cv2.imwrite(gray_frame_path, gray_frame)

    gray_frame = run_object_detection_and_depth_estimation(image_path=gray_frame_path)

    os.remove(gray_frame_path)


    # Optionally: Process the frame further based on crash detection (e.g., convert to grayscale, annotate)
    # For simplicity, we'll just convert to grayscale
    # gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_BGR2GRAY)

    # Save the processed (or original) frame to a file with a serial name
    filename = f"{save_folder}/{frame_counter}.jpg"


    ###########################################################################################################################################
    cv2.imwrite(filename, gray_frame)#gray_frame if crash_detected else frame)
    ###########################################################################################################################################


    # Increment the frame counter for the next frame
    frame_counter += 1
    ###########################################################################################################################################

    # Return the processed frame or some response indicating crash status
    _, buffer = cv2.imencode('.jpg', gray_frame)#gray_frame if crash_detected else frame)
    processed_frame_bytes = buffer.tobytes()
    ###########################################################################################################################################

    return StreamingResponse(BytesIO(processed_frame_bytes), media_type="image/jpeg")



####################################################################################################################

# from rasberry1 import run_object_detection_and_depth_estimation


# import os
# import cv2
# import numpy as np
# from datetime import datetime, timedelta
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import StreamingResponse
# from io import BytesIO
# import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
# from email.mime.image import MIMEImage

# app = FastAPI()

# # Assume other initializations and imports are done above

# # Global variable to track the last email sent time
# last_email_sent_time = datetime.min

# def can_send_email():
#     global last_email_sent_time
#     return datetime.now() - last_email_sent_time >= timedelta(minutes=1)

# def send_email_with_frame(frame, crash_probability, receiver_email):
#     global last_email_sent_time
#     sender_email = 'priyam22rr@gmail.com'
#     password = 'mlonvvatlfnilogu'  # Use your generated app password here

#     subject = 'Crash Detection Alert'
#     body = f'Crash probability: {crash_probability}'

#     msg = MIMEMultipart()
#     msg['From'] = sender_email
#     msg['To'] = receiver_email
#     msg['Subject'] = subject

#     msg.attach(MIMEText(body, 'plain'))

#     # Convert frame to JPEG Bytes and attach
#     frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
#     image = MIMEImage(frame_bytes, name='frame.jpg')
#     msg.attach(image)

#     with smtplib.SMTP('smtp.gmail.com', 587) as server:
#         server.starttls()
#         server.login(sender_email, password)
#         server.send_message(msg)

#     last_email_sent_time = datetime.now()

# @app.post("/api/process_frame/")
# async def process_frame_api(file: UploadFile = File(...)):
#     # Read and process the frame as before

#     global frame_counter

#     # Read the uploaded frame
#     contents = await file.read()

#     # Decode the frame
#     nparr = np.frombuffer(contents, np.uint8)
#     frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     # Call the adjusted function to process the frame and predict crash
#         ###########################################################################################################################################
#     crash_detected, crash_probability = extract_frame_and_process(frame, crash_detection_model)
#     print(crash_detected, crash_probability)
#         ###########################################################################################################################################


#     gray_frame = frame

#     gray_frame_path = "gray_frame.jpg"
#     cv2.imwrite(gray_frame_path, gray_frame)

#     gray_frame = run_object_detection_and_depth_estimation(image_path=gray_frame_path)

#     os.remove(gray_frame_path)


#     # After processing the frame and determining crash probability
#     if can_send_email():
#         # Replace 'receiver_email@example.com' with the actual recipient's email
#         send_email_with_frame(gray_frame, crash_probability, 'vipul0592bhatia@gmail.com')


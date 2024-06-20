import cv2
import onnx
import onnxruntime
import numpy as np
from YOLOv7 import YOLOv7
from utils import draw_detections
import math
from moviepy import editor


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


# def region_of_interest(img, vertices):
#     """
#     Applies an image mask.

#     Only keeps the region of the image defined by the polygon
#     formed from `vertices`. The rest of the image is set to black.
#     `vertices` should be a numpy array of integer points.
#     """
#     # defining a blank mask to start with
#     mask = np.zeros_like(img)

#     # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
#     if len(img.shape) > 2:
#         channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
#         ignore_mask_color = (255,) * channel_count
#     else:
#         ignore_mask_color = 255

#     # filling pixels inside the polygon defined by "vertices" with the fill color
#     cv2.fillPoly(mask, vertices, ignore_mask_color)

#     # returning the image only where mask pixels are nonzero
#     masked_image = cv2.bitwise_and(img, mask)

#     # cv2.imshow("masked_image", masked_image)
#     # cv2.waitKey(0)

#     return remove_horizontal_noice(masked_image)

#     return masked_image


# def remove_horizontal_noice(img, angle_range=(30, 50)):
#     lines = cv2.HoughLinesP(
#         img, rho=1, theta=np.pi / 180, threshold=20, minLineLength=20, maxLineGap=180
#     )

#     mask = np.zeros_like(img)

#     # Check if lines are detected
#     if lines is not None:
#         # Filter lines based on their angles
#         for line in lines:
#             x1, y1, x2, y2 = line[0]

#             angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

#             print(angle)

#             if angle_range[0] <= abs(angle) <= angle_range[1]:
#                 cv2.line(mask, (x1, y1), (x2, y2), 255, 2)
#                 # cv2.imshow("Lane Detection", mask)
#                 # cv2.waitKey(0)

#         # Inpaint the mask
#         inpainted_image = cv2.inpaint(mask, img, 3, cv2.INPAINT_TELEA)

#         # cv2.imshow("inpainted_image", inpainted_image)
#         # cv2.waitKey(0)

#         return inpainted_image

#     else:
#         # If no lines are detected, return the original image
#         return img


#     cv2.imshow("inpainted_image", inpainted_image)
#     cv2.waitKey(0)


# def draw_lines(img, lines, color=[255, 0, 0], thickness=10, expansion_percentage=30):
#     """
#     This function draws `lines` with `color` and `thickness`.
#     Lines are drawn on the image inplace (mutates the image).
#     It also expands the lines by the specified percentage.

#     Args:
#       img: The image on which to draw the lines.
#       lines: A list of lines, where each line is represented by the coordinates (x1, y1, x2, y2).
#       color: The color of the lines.
#       thickness: The thickness of the lines.
#       expansion_percentage: The percentage by which to expand the lines.
#     """
#     for line in lines:
#         for x1, y1, x2, y2 in line:
#             # Calculate the slope of the line
#             slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')

#             # Calculate expansion distance
#             expansion_distance = int((x2 - x1) * expansion_percentage / 100)

#             # Expand the lines in both directions
#             expanded_x1 = x1 - expansion_distance
#             expanded_y1 = int(y1 - slope * expansion_distance)

#             expanded_x2 = x2 + expansion_distance
#             expanded_y2 = int(y2 + slope * expansion_distance)

#             # Draw the expanded lines
#             cv2.line(img, (expanded_x1, expanded_y1), (expanded_x2, expanded_y2), color, thickness)



# def slope_lines(image, lines):
#     img = image.copy()
#     poly_vertices = []
#     order = [0, 1, 3, 2]

#     left_lines = []  # Like /
#     right_lines = []  # Like \
    
#     for line in lines:
#         for x1, y1, x2, y2 in line:
#             if x1 == x2:
#                 pass  # Vertical Lines
#             else:
#                 m = (y2 - y1) / (x2 - x1)
#                 c = y1 - m * x1

#                 if m < 0:
#                     left_lines.append((m, c))
#                 elif m >= 0:
#                     right_lines.append((m, c))

#     left_line = np.mean(left_lines, axis=0)
#     right_line = np.mean(right_lines, axis=0)

#     if len(left_lines) == 0 or len(right_lines) == 0:
#         return image

#     for slope, intercept in [left_line, right_line]:
#         # Calculate line extension coordinates
#         y1 = int(img.shape[0])  # Start at the bottom of the image
#         y2 = int(img.shape[0] * 0.6)  # Extend the line up to 60% of the image height

#         # Calculate corresponding x values using the line equation x = (y - c) / m
#         x1 = int((y1 - intercept) / slope)
#         x2 = int((y2 - intercept) / slope)

#         # Draw the extended lines
#         cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 10)

#         # Store the extended line coordinates for polygon filling
#         poly_vertices.append((x1, y1))
#         poly_vertices.append((x2, y2))

#     poly_vertices = [poly_vertices[i] for i in order]

#     vertices = np.array([poly_vertices], dtype=np.int32)
#     cv2.polylines(img, vertices, isClosed=True, color=[0, 0, 255], thickness=10)
    
#     # Fill the polygon formed by the extended lines
#     cv2.fillPoly(img, pts=np.array([poly_vertices], "int32"), color=(0, 255, 0))

#     return cv2.addWeighted(image, 0.7, img, 0.4, 0.0), poly_vertices



# def line_intersection(line1, line2):
#     """
#     This function checks if two line segments intersect and returns the intersection point.

#     Args:
#       line1: A numpy array of shape (2, 2) representing the start and end points of the first line segment.
#       line2: A numpy array of shape (2, 2) representing the start and end points of the second line segment.

#     Returns:
#       A numpy array of shape (2,) representing the intersection point of the two line segments, or None if they do not intersect.
#     """
#     x2, y2 = line1.flatten()
#     x1 = x2
#     y1 = 0
#     x4, y4 = line2.flatten()
#     x3 = x4
#     y3 = 0

#     denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
#     if denominator == 0:
#         return None  # lines are parallel

#     t = ((x3 - x4) * (y1 - y3) - (y3 - y4) * (x1 - x3)) / denominator
#     u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator

#     if 0 <= t <= 1 and 0 <= u <= 1:
#         return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])
#     else:
#         return None


# def shorten_lines(line1, line2, intersection_point):
#     """
#     This function shortens two intersecting line segments from the intersection point.

#     Args:
#       line1: A numpy array of shape (2, 2) representing the start and end points of the first line segment.
#       line2: A numpy array of shape (2, 2) representing the start and end points of the second line segment.
#       intersection_point: A numpy array of shape (2,) representing the intersection point of the two line segments.

#     Returns:
#       Two numpy arrays of shape (2, 2) representing the shortened line segments.
#     """
#     # Get the direction vectors of the lines
#     direction1 = line1[1] - line1[0]
#     direction2 = line2[1] - line2[0]

#     # Project the intersection point onto each line
#     projected_point1 = (
#         line1[0]
#         + np.dot(intersection_point - line1[0], direction1)
#         / np.dot(direction1, direction1)
#         * direction1
#     )
#     projected_point2 = (
#         line2[0]
#         + np.dot(intersection_point - line2[0], direction2)
#         / np.dot(direction2, direction2)
#         * direction2
#     )

#     # Shorten the lines by moving their end points to the projected points
#     shortened_line1 = np.array([line1[0], projected_point1])
#     shortened_line2 = np.array([line2[0], projected_point2])

#     return shortened_line1, shortened_line2


# def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
#     """
#     `img` should be the output of a Canny transform.

#     Returns an image with hough lines drawn.
#     """
#     lines = cv2.HoughLinesP(
#         img,
#         rho,
#         theta,
#         threshold,
#         np.array([]),
#         minLineLength=min_line_len,
#         maxLineGap=max_line_gap,
#     )
#     line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
#     poly_vertices = []  # Initialize poly_vertices here

#     # draw_lines(line_img, lines)
#     if lines is not None:
#         line_img, poly_vertices = slope_lines(line_img, lines)

#     return line_img, poly_vertices



# # Python 3 has support for cool math symbols.


# def weighted_img(img, initial_img, α=0.1, β=1.0, γ=0.0):
#     """
#     `img` is the output of the hough_lines(), An image with lines drawn on it.
#     Should be a blank image (all black) with lines drawn on it.

#     `initial_img` should be the image before any processing.

#     The result image is computed as follows:

#     initial_img * α + img * β + γ
#     NOTE: initial_img and img must be the same shape!
#     """
#     lines_edges = cv2.addWeighted(initial_img, α, img, β, γ)
#     # lines_edges = cv2.polylines(lines_edges,get_vertices(img), True, (0,0,255), 10)
#     return lines_edges


# def get_vertices(image):
#     rows, cols = image.shape[:2]
#     bottom_left = [cols * 0.15, rows]
#     top_left = [cols * 0.45, rows * 0.6]
#     bottom_right = [cols * 0.95, rows]
#     top_right = [cols * 0.55, rows * 0.6]

#     ver = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

#     return ver


# # Lane finding Pipeline
# def lane_finding_pipeline(image):
#     print("Entering lane_finding_pipeline")
#     # cv2.imshow("image", image)
#     # cv2.waitKey(0)

#     # Grayscale
#     gray_img = grayscale(image)

#     # cv2.imshow("grayscale", gray_img)
#     # cv2.waitKey(0)

#     # Gaussian Smoothing
#     smoothed_img = gaussian_blur(img=gray_img, kernel_size=5)

#     # cv2.imshow("smoothed_img", smoothed_img)
#     # cv2.waitKey(0)

#     # Canny Edge Detection
#     canny_img = canny(img=smoothed_img, low_threshold=180, high_threshold=240)

#     # cv2.imshow("canny_img", canny_img)
#     # cv2.waitKey(0)

#     # Masked Image Within a Polygon
#     masked_img = region_of_interest(img=canny_img, vertices=get_vertices(canny_img))

#     # cv2.imshow("masked_img", masked_img)
#     # cv2.waitKey(0)

#     # Hough Transform Lines
#     houghed_lines, poly_vertices = hough_lines(
#         img=masked_img,
#         rho=1,
#         theta=np.pi / 180,
#         threshold=20,
#         min_line_len=20,
#         max_line_gap=180,
#     )

#     # cv2.imshow("houghed_lines", houghed_lines)
#     # cv2.waitKey(0)

#     # Draw lines on edges
#     output = weighted_img(img=houghed_lines, initial_img=image, α=0.8, β=1.0, γ=0.0)
    

#     # cv2.imshow('Lane weighted_img', output)
#     # cv2.waitKey(0)
#     print("After lane_finding_pipeline call")

#     # cv2.imshow('Combined Object Detection and Depth Estimation', output)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()


#     return output, poly_vertices


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

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# Run the function with the path to your input image
# run_object_detection_and_depth_estimation(image_path='road.jpeg')




# from fastapi.middleware.cors import CORSMiddleware
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import StreamingResponse
# from pydantic import BaseModel
# from io import BytesIO
# import cv2
# import numpy as np
# import os
# from datetime import datetime

# app = FastAPI()

# origins = ["http://localhost:3000"]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Create a folder to save the frames
# save_folder = "saved_frames"
# os.makedirs(save_folder, exist_ok=True)

# # Counter for serial naming
# frame_counter = 1

# # Endpoint to process the received frame using an OpenCV model
# @app.post("/api/process_frame/")
# async def process_frame(file: UploadFile = File(...)):
#     global frame_counter

#     # Read the uploaded frame
#     contents = await file.read()

#     # Decode the frame
#     nparr = np.frombuffer(contents, np.uint8)
#     frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     # Perform your OpenCV model processing on the frame
#     # ...

#     # Example: Convert the frame to grayscale
#     gray_frame = frame

#     gray_frame_path = "gray_frame.jpg"
#     cv2.imwrite(gray_frame_path, gray_frame)

#     gray_frame = run_object_detection_and_depth_estimation(image_path=gray_frame_path)

#     os.remove(gray_frame_path)

#     # Save the processed frame to a file with a serial name
#     filename = f"{save_folder}/{frame_counter}.jpg"
#     cv2.imwrite(filename, gray_frame)

#     # Increment the frame counter for the next frame
#     frame_counter += 1

#     # Return the processed frame
#     _, buffer = cv2.imencode('.jpg', gray_frame)
#     processed_frame_bytes = buffer.tobytes()

#     return StreamingResponse(BytesIO(processed_frame_bytes), media_type="image/jpeg")










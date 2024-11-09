from src.object_detector.yolov3 import YoloPeopleDetector
from src.object_detector.postprocessor import PostProcessor
from src.visualization.visualizer import CameraViz
import cv2
import argparse
import os
import sys
import numpy as np

# Initialize YOLO network, postprocessor, and visualization mode
net = YoloPeopleDetector()
net.load_network()

# Process inputs
parser = argparse.ArgumentParser(description='Run social distancing meter')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()
winName = 'predicted people'
cv2.namedWindow(winName, cv2.WINDOW_NORMAL)

outputFile = "yolo_out_py.avi"
if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print(f"Input image file {args.image} doesn't exist")
        sys.exit(1)
    cap = cv2.VideoCapture(args.image)
    outputFile = args.image[:-4] + '_yolo_out_py.jpg'
elif (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print(f"Input video file {args.video} doesn't exist")
        sys.exit(1)
    cap = cv2.VideoCapture(args.video)
    outputFile = args.video[:-4] + '_yolo_out_py.avi'
else:
    # Webcam input
    cap = cv2.VideoCapture(0)

# Get the video writer initialized to save the output video
if not args.image:
    vid_writer = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, 
                                 (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# Dummy data for the visualization (replace with actual data in your project)
viofeed = np.random.randint(1, 100, 10)  # Example data
nonviofeed = np.random.randint(1, 100, 10)  # Example data
sevidx = np.random.rand(10)  # Example data
violocationsx = np.random.randint(0, 100, 10)  # Example data
violocationsy = np.random.randint(0, 100, 10)  # Example data

while True:
    hasFrame, frame = cap.read()
    if not hasFrame:
        print("Done processing!")
        break

    # Make sure frame is valid and has appropriate dimensions
    if frame is None or frame.shape[1] < 1920:
        print("Frame dimension is too small to fit the dashboard layout.")
        break

    # Run YOLO object detection
    outs = net.predict(frame)
    pp = PostProcessor()
    indices, boxes, ids, confs, centers = pp.process_preds(frame, outs)

    # Collect dynamic data for visualization (real-time updates)
    viofeed = np.random.randint(1, 100, 10)  # Update this with actual dynamic data
    nonviofeed = np.random.randint(1, 100, 10)  # Update with actual data
    sevidx = np.random.rand(10)  # Update severity index dynamically
    violocationsx = np.random.randint(0, 100, 10)  # Update with actual data
    violocationsy = np.random.randint(0, 100, 10)  # Update with actual data

    # Create the CameraViz object with the updated data
    visualizer = CameraViz(indices, frame, ids, confs, boxes, centers)

    # Draw predictions and dashboard on the frame
    output_frame = visualizer.draw_pred(frame, viofeed, nonviofeed, sevidx, violocationsx, violocationsy)

    # Display the frame with dashboard
    cv2.imshow("Analyzed Video with Dashboard", output_frame)

    # Write the output frame to the video file
    if not args.image:
        vid_writer.write(output_frame.astype(np.uint8))

    # Break if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

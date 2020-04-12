# USAGE

# python detect_face_caffe_resnet.py --prototxt ..\caffe\deploy.prototxt.txt --model ..\caffe\res10_300x300_ssd_iter_140000.caffemodel
# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import socket
import imutils
import time
import cv2
import os

HOSTNAME = socket.gethostname()
def check_path(path):            #function to confirm whether the given path exists or not
    dir = os.path.dirname(path)  #if it doesn't exist this function will create
    if not os.path.exists(dir):
        os.makedirs(dir)
face_id = 1  # For each person,there will be one face id
count = 0    # Initialize sample face image

check_path("dataset/" + HOSTNAME + "/")

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(os.path.join(os.getcwd(), 'caffe', 'deploy.prototxt.txt'), os.path.join(os.getcwd(), 'caffe', 'res10_300x300_ssd_iter_140000.caffemodel'))

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = 0
frame_num = 0
# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=640)
	start_time = time.time()
	frame_num = frame_num + 1
	# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
 
	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence < 0.5:
			continue

		# compute the (x, y)-coordinates of the bounding box for the
		# object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		count += 1               # Increment face image

		cv2.imwrite("dataset/"+ HOSTNAME +"/user." + str(face_id) + '.' + str(count) + ".jpg", frame)
		# draw the bounding box of the face along with the associated
		# probability
		text = "{:.2f}%".format(confidence * 100)
		# y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
		# cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

		end_time = time.time()
		fps = fps * 0.91 + 1/(end_time - start_time) * 0.1
		start_time = end_time
		frame_info = 'Frame: {0}, FPS: {1:.2f}'.format(frame_num, fps)
		cv2.putText(frame, frame_info, (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
	elif count>100:                                     # If image taken reach 100, stop taking video
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
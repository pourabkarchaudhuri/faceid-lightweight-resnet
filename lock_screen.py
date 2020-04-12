# USAGE
# python recognize_video.py --detector face_detection_model \
#	--embedding-model openface_nn4.small2.v1.t7 \
#	--recognizer output/recognizer.pickle \
#	--le output/le.pickle

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
# import pyautogui
import imutils
import pickle
import ctypes
import time
import cv2
import os
import sys
import logging

CURRENT_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(os.getcwd(), 'output')
EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, 'embeddings.pickle')
RECOGNIZER_PATH = os.path.join(OUTPUT_DIR, 'recognizer.pickle')
LABEL_ENCODER_PATH = os.path.join(OUTPUT_DIR, 'le.pickle')
EMBEDDING_MODEL_PATH = os.path.join(CURRENT_DIR, 'openface_nn4.small2.v1.t7')
LOG_FILE_PATH = os.path.join(CURRENT_DIR, 'log', 'app.log')
GLOBAL_FACE_DETECTION_THRESHOLD = 0.5
GLOBAL_TRIGGER_DELAY = 30
GLOBAL_LOGGER_DELAY = GLOBAL_TRIGGER_DELAY/2
VIDEO_SOURCE=0

counter_correct = 0  #counter variable to count number of times loop runs
counter_wrong = 0
no_of_faces= 0
fps_counter = 0
frame_num = 0
frame_num_trigger = 0

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.join(os.getcwd(), 'caffe', 'deploy.prototxt.txt')
modelPath = os.path.join(os.getcwd(), 'caffe', 'res10_300x300_ssd_iter_140000.caffemodel')
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(EMBEDDING_MODEL_PATH)

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(RECOGNIZER_PATH, "rb").read())
le = pickle.loads(open(LABEL_ENCODER_PATH, "rb").read())

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=VIDEO_SOURCE).start()
time.sleep(2.0)

# start the FPS throughput estimator
fps = FPS().start()

# loop over frames from the video file stream
while True:
    
    frame = vs.read()
    start_time = time.time()
    frame_num += 1
   
    # resize the frame to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image
    # dimensions
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > GLOBAL_FACE_DETECTION_THRESHOLD:
            no_of_faces += 1

            # compute the (x, y)-coordinates of the bounding box for
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            # draw the bounding box of the face along with the
            # associated probability
            text = "{}: {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            
            
            # frame_num_temp += 1
            if(name == 'unknown'):
                # print("Stranger Detected")
                cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(frame, 'Stranger Frame Count: {0}'.format(counter_wrong), (10, frame.shape[0]-80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                counter_wrong += 1
                counter_correct = 0
            else:
                # print("Host Detected")
                cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 2)
                cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, 'Host Frame Count: {0}'.format(counter_correct), (10, frame.shape[0]-80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                counter_correct += 1
                counter_wrong = 0
            
            # Use set counters to act on triggers
            if(counter_wrong > GLOBAL_TRIGGER_DELAY): # Stranger detected in set consecutive frames, Lock screen
                print("Stranger Detected")
                if(counter_wrong == GLOBAL_LOGGER_DELAY):
                    print("Logging event")
                    logging.basicConfig(filename=LOG_FILE_PATH, filemode='w', format='%(name)s - %(levelname)s - %(message)s')
                    logging.warning('STRANGER DETECTED', str(no_of_faces))
                vs.stop()
                cv2.destroyAllWindows()
                ctypes.windll.user32.LockWorkStation()
                sys.exit()
                # Remember to log these images and text data

           
    # update the FPS counter
    fps.update()
    cv2.putText(frame, 'Face Count: {0}'.format(no_of_faces), (10, frame.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    # print("Frame Number Temp {}".format(frame_num_trigger))
    if(no_of_faces==0 or no_of_faces>1):
        frame_num_trigger += 1
        # print("Number of faces : {0}".format(no_of_faces))
        if(frame_num_trigger > GLOBAL_TRIGGER_DELAY):
            if(frame_num_trigger == GLOBAL_LOGGER_DELAY):
                print("Logging event")
                logging.basicConfig(filename=LOG_FILE_PATH, filemode='w', format='%(name)s - %(levelname)s - %(message)s')
                logging.warning('%s FACES DETECTED', str(no_of_faces))

            print("Locking screen. No of faces:{0}".format(no_of_faces))
            vs.stop()
            cv2.destroyAllWindows()
            ctypes.windll.user32.LockWorkStation()
            sys.exit()
    
    else:
        frame_num_trigger = 0

    # Reset number of faces
    no_of_faces = 0
    
    end_time = time.time()
    fps_counter = fps_counter * 0.91 + 1/(end_time - start_time) * 0.1
    start_time = end_time
    frame_info = 'Frame: {0}, FPS: {1:.2f}'.format(frame_num, fps_counter)
    
    cv2.putText(frame, frame_info, (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # frame_num_trigger += 1
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()


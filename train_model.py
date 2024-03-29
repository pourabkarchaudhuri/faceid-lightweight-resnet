# USAGE
# python train_model.py --embeddings output/embeddings.pickle \
#	--recognizer output/recognizer.pickle --le output/le.pickle

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
import os

# construct the argument parser and parse the arguments

EMBEDDINGS_PATH = os.path.join(os.getcwd(), 'output', 'embeddings.pickle')
RECOGNIZER_PATH = os.path.join(os.getcwd(), 'output', 'recognizer.pickle')
LABEL_ENCODER_PATH = os.path.join(os.getcwd(), 'output', 'le.pickle')
# load the face embeddings
print("[INFO] loading face embeddings...")

data = pickle.loads(open(EMBEDDINGS_PATH, "rb").read())

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# write the actual face recognition model to disk
f = open(RECOGNIZER_PATH, "wb")
f.write(pickle.dumps(recognizer))
f.close()

# write the label encoder to disk
f = open(LABEL_ENCODER_PATH, "wb")
f.write(pickle.dumps(le))
f.close()
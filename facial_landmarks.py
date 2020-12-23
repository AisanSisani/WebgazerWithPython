# import the necessary packages
import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import pyautogui

'''
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())
'''


def face_landmark(image):
	# load the input image, resize it, and convert it to grayscale
	## image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale image
	rects = detector(gray, 1)
	landmarks = []
	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		landmarks.append(shape)

		# convert dlib's rectangle to a OpenCV-style bounding box
		# [i.e., (x, y, w, h)], then draw the face bounding box
		(x, y, w, h) = face_utils.rect_to_bb(rect)
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

		# show the face number
		cv2.putText(image, "Face #{}".format(i + 1), (x - 8, y - 8),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

		mousex, mousey = pyautogui.position()
		cv2.putText(image, "{}, {}".format(mousex, mousey), (10,10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		for (x, y) in shape:
			cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

	# show the output image with the face detections + facial landmarks
	cv2.imshow("Output", image)
	return landmarks, mousex, mousey

def crop_eye(image, landmarks):
	(x, y, w, h) = cv2.boundingRect(landmarks)
	eye = image[y - 5:y + h + 5, x - 5:x + w + 5]
	eye = cv2.resize(eye, (60, 30), interpolation=cv2.INTER_AREA)
	return eye

def show_eyes(image, landmarks):
	landmarks = landmarks[0]

	#left eyes
	i, j = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
	left_landmarks = np.array([landmarks[i:j]])
	left_eye = crop_eye(image, left_landmarks)

	i, j = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
	right_landmarks = np.array([landmarks[i:j]])
	right_eye = crop_eye(image, right_landmarks)

	# show the particular face part
	cv2.imshow("left", left_eye)
	cv2.imshow("right", right_eye)

	return left_eye, right_eye

#------------------------------------main----------------------------------
predictor_path = "shape_predictor_68_face_landmarks.dat" # =args["shape_predictor"]
image_path = "image.jpg" #=args["image"]

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

cap = cv2.VideoCapture(0)
i=0
coordinates = []

cv2.namedWindow("full_window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("full_window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

import ctypes
## get Screen Size
user32 = ctypes.windll.user32
width, height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
blank_image = np.ones((height, width, 3), np.uint8)

while cap.isOpened():
	_, img = cap.read()

	landmarks, mousex, mousey = face_landmark(img)
	left, right = show_eyes(img, landmarks)


	cv2.imwrite("pictures/{}_left.jpg".format(i), left)
	cv2.imwrite("pictures/{}_right.jpg".format(i), right)
	i=i+1

	cv2.circle(blank_image, (mousex, mousey), 1, (0, 0, 255), -1)
	cv2.imshow("full_window", blank_image)

	coordinates.append((i, (mousex, mousey)))
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()

import pickle
with open("pictures/coordinates.txt", "wb") as fp:   #Pickling
	pickle.dump(coordinates, fp)
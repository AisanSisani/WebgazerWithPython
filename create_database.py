# import the necessary packages
import face_utils
import numpy as np
import dlib
import cv2
import pyautogui


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

	return landmarks

def crop_eye(image, landmarks):
	(x, y, w, h) = cv2.boundingRect(landmarks)
	eye = image[y - 5:y + h + 5, x - 8:x + w + 8]
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

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

cap = cv2.VideoCapture(0)
i=0
coordinates = []

while cap.isOpened():
	_, img = cap.read()
	mousex, mousey = pyautogui.position()

	landmarks = face_landmark(img)
	left, right = show_eyes(img, landmarks)

	cv2.imwrite("pictures/head/{}_left.jpg".format(i), left)
	cv2.imwrite("pictures/head/{}_right.jpg".format(i), right)
	coordinates.append((i, (mousex, mousey)))
	i=i+1


	cv2.imshow("camera", img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()

import pickle
with open("pictures/head/coordinates.txt", "wb") as fp:   #Pickling
	pickle.dump(coordinates, fp)
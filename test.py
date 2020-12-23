# import the necessary packages
import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import pyautogui
import ctypes

def face_landmark(image):
	# load the input image, resize it, and convert it to grayscale
	## image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    landmarks = []
	# loop over the face detections
    for (i, rect) in enumerate(rects):
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

#------------------------------------------------------main------------------------------------
predictor_path = "shape_predictor_68_face_landmarks.dat" # =args["shape_predictor"]
image_path = "image.jpg" #=args["image"]

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

cap = cv2.VideoCapture(0)

## get Screen Size
user32 = ctypes.windll.user32
width, height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
blank_image = np.ones((height, width, 3), np.uint8)

cv2.namedWindow("full_window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("full_window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

import pickle
filename = 'models/first_modelx.sav'
modelx = pickle.load(open(filename, 'rb'))
filename = 'models/first_modely.sav'
modely = pickle.load(open(filename, 'rb'))

while cap.isOpened():
    _, img = cap.read()
    mousex, mousey = pyautogui.position()

    landmarks = face_landmark(img)
    limg, rimg = show_eyes(img, landmarks)

    limg = cv2.cvtColor(limg, cv2.COLOR_BGR2GRAY)
    limg = np.array(limg)
    limg = limg.flatten()
    rimg = cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY)
    rimg = np.array(rimg)
    rimg = rimg.flatten()
    img = np.concatenate((limg, rimg))

    predx = round(modelx.predict([img])[0])
    predy = round(modely.predict([img])[0])
    
    cv2.circle(blank_image, (predx, predy), 1, (0, 0, 255), -1)
    cv2.imshow("full_window", blank_image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
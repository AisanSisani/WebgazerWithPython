# Created: 25/02/2019
# Copyright: (c) PNU 2019
#--------------------------------------------------------------------
import cv2
import ctypes

## get Screen Size
user32 = ctypes.windll.user32
screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


def showImage(filename):
     W,H = screensize
     oriimg = cv2.imread(filename)
     height, width, depth = oriimg.shape

     scaleWidth = float(W)/float(width)
     scaleHeight = float(H)/float(height)
     if scaleHeight>scaleWidth:
          imgScale = scaleWidth
     else:
          imgScale = scaleHeight

     newX,newY = oriimg.shape[1]*imgScale, oriimg.shape[0]*imgScale
     newimg = cv2.resize(oriimg,(int(newX),int(newY)))
     cv2.imshow("fit image to screen by CV2",newimg)
     cv2.waitKey(0)

if __name__ == '__main__':
     filename = 'right_97.jpg'
     showImage(filename)
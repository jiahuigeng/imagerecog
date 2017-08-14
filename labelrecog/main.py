import numpy as np
import cv2
'''
im_gray = cv2.imread('2.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)
thresh = 130
im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
im_wb=255-im_bw
cv2.imshow('image',im_wb)
cv2.imwrite('messigray.png',im_wb)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))


closed = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)

res = cv2.resize(img,None,fx=5, fy=5, interpolation = cv2.INTER_CUBIC)
kernel = np.ones((5,5),np.uint8)
dilation = cv2.erode(res,kernel,iterations = 3)
'''

im_gray = cv2.imread('2.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE)



cv2.imshow("Close",im_gray);
cv2.imwrite('closed.png',im_gray)
cv2.waitKey(0)

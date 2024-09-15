import cv2
import numpy as np
from iofile import *

# Load the two images
img1 = cv2.imread('result/ascii2.jpg')  
img2 = cv2.imread('result/edge_ascii_2.jpg')  


gray1 = desat_graysc(img1,True)
gray2 = desat_graysc(img2,True)

# Threshold Image 2 to create a mask of edges
_, mask = cv2.threshold(gray2, 50, 255, cv2.THRESH_BINARY)

# Invert the mask to get the non-edge areas
mask_inv = cv2.bitwise_not(mask)

# Use the mask to keep edges from Image 2
edges_from_img2 = cv2.bitwise_and(gray2, gray2, mask=mask)

# Use the inverted mask to keep the non-edge parts from Image 1
filled_from_img1 = cv2.bitwise_and(gray1, gray1, mask=mask_inv)

# Combine the edges and filled areas
combined_image = cv2.add(edges_from_img2, filled_from_img1)

# Display the result
cv2.imshow('Combined ASCII Art', combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('result/combined_ascii.jpg', combined_image)
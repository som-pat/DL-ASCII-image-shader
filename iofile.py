import cv2
import numpy as np
import math


def display(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def image_sharpen(img): #Image sharpening kernel
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) / 1
    img = cv2.filter2D(img, cv2.CV_8U, kernel)
    display(img)
    return img

def gradient_direction(img):
    Sx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3,scale=1,delta=0,borderType=cv2.BORDER_DEFAULT)
    Sy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3,scale=1,delta=0,borderType=cv2.BORDER_DEFAULT)

    grad_theta = np.atan2(Sy,Sx)/(2*np.pi)    
    theta_normalise = np.add(grad_theta,0.5,where = grad_theta!=0 )
    theta_image = (theta_normalise*255).astype(np.uint8)

    return theta_image




def extended_sobel(img):
    grad_x = cv2.Sobel(img, cv2.CV_16S, 1, 0,ksize=3,scale=1,delta=0,borderType=cv2.BORDER_DEFAULT) #(x:1,y:0) 16bit gradient change along X axis
    grad_y = cv2.Sobel(img, cv2.CV_16S, 0, 1,ksize=3,scale=1,delta=0,borderType=cv2.BORDER_DEFAULT) #(x:0,y:1) 16bit gradient change along y axis
    

    abs_grad_x = cv2.convertScaleAbs(grad_x) #Scales, calculates absolute values, and converts the result to 8-bit.
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0) #Calculates the weighted sum of two arrays.    

    output = grad.astype(np.uint8)
    print('sobel')
    
    return output



def difference_of_Gaussian(img, sigma1, sigma2):
    grad1 = cv2.GaussianBlur(img,(0,0),sigma1)
    grad2 = cv2.GaussianBlur(img,(0,0),sigma2)
    DoG_img = cv2.subtract(grad1, grad2) 
    return DoG_img


def enhance_contrast(img):
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    en_img = clahe.apply(img)
    return en_img

def canny(img):
    edge_img = cv2.Canny(img,100,200)
    return edge_img

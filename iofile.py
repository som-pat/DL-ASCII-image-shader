import cv2
import numpy as np
import math


def display(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def desat_graysc(img,cond):
    if cond==True:
        grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print('grayscale_image')
        display(grayscale_image)
        return grayscale_image
    
    elif cond ==False:
        desaturated_image = np.mean(img, axis=2).astype(np.uint8)
        display(desaturated_image)
        return desaturated_image

#  desaturated_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS) #np.mean(img, axis=2).astype(np.uint8)
#         print('HLS method')
#         display(desaturated_image)
#         desaturated_image[:,:,2] = desaturated_image[:,:, 2] *0.2
#         display(desaturated_image)
#         desaturated_image = cv2.cvtColor(desaturated_image, cv2.COLOR_HLS2BGR) 
#         print('desaturated_image')
#         display(desaturated_image)
#         desaturated_image = np.mean(img, axis=2).astype(np.uint8)
#         display(desaturated_image)

def hsv_val(img):
    hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV_FULL)
    h, s, val_img = cv2.split(hsv_img)
    val_img = cv2.multiply(val_img, 1.5)
    display(val_img)
    val_img =  np.clip(val_img, 0, 255).astype(np.uint8)
    display(val_img)
    filter_val_img = cv2.merge([h, s, val_img])
    display(filter_val_img)
    filter_val_img = cv2.cvtColor( filter_val_img, cv2.COLOR_HSV2BGR)
    display(filter_val_img)

    return filter_val_img



def threshold_saturation(img):
    hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV_FULL)
    _, saturate_img, _ = cv2.split(hsv_img)
    saturate_img = cv2.multiply(saturate_img, 1.5)
    saturate_img = np.clip(saturate_img, 0, 255).astype(np.uint8)
    
    print('sat_thres')
    display(saturate_img)

    return saturate_img

def satval_gradient(img):
    hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV_FULL)
    h, saturate_img, val_img = cv2.split(hsv_img)

    val_img = cv2.multiply(val_img, 1.5)
    val_img =  np.clip(val_img, 0, 255).astype(np.uint8)

    saturate_img = cv2.multiply(saturate_img, 2.5)
    saturate_img = np.clip(saturate_img, 0, 255).astype(np.uint8)

    filter_img = cv2.merge([h, saturate_img, val_img])
    filter_img = cv2.cvtColor( filter_img, cv2.COLOR_HSV2BGR)
    display(filter_img)

    return filter_img




def image_sharpen(img): #Image sharpening kernel
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) / 1
    img = cv2.filter2D(img, cv2.CV_8U, kernel)
    display(img)
    return img

def gradient_direction(img):
    Sx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    Sy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)    

    grad_theta = np.atan2(Sy,Sx)/(2*np.pi)
    grad_theta[grad_theta <= -0.4 ] = 0
    print('grad_theta',grad_theta)      
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



def morph_filter(img):
    thresh = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)[1]
    display(thresh)

    # morphology edgeout = dilated_mask - mask
    # morphology dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    dilate = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
    display(dilate)
    # get absolute difference between dilate and thresh
    diff = cv2.absdiff(dilate, thresh)
    display(diff)
    edges = 255 -diff
    display(edges)

def orientation_map(mag, ori, thresh=1.0):
    # Create an empty image for the orientation map
    ori_map = np.zeros((ori.shape[0], ori.shape[1], 3), dtype=np.uint8)
    
    # Define color mappings for specific orientation ranges
    red = (0, 0, 255)
    cyan = (255, 255, 0)
    green = (0, 255, 0)
    yellow = (0, 255, 255)
    
    # Iterate through each pixel to assign colors based on orientation
    for i in range(mag.shape[0]):
        for j in range(mag.shape[1]):
            if mag[i, j] > thresh:
                angle = ori[i, j]
                if angle < 90.0:
                    ori_map[i, j] = red
                elif 90.0 <= angle < 180.0:
                    ori_map[i, j] = cyan
                elif 180.0 <= angle < 270.0:
                    ori_map[i, j] = green
                elif 270.0 <= angle < 360.0:
                    ori_map[i, j] = yellow
    
    display(ori_map)
    return ori_map


def sobel_grad_orient(img):
    Sx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    Sy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    # Calculate magnitude and orientation of gradients
    mag = cv2.magnitude(Sx, Sy)
    display(mag)
    ori = cv2.phase(Sx, Sy, angleInDegrees=True)
    display(ori)
    ori_map = orientation_map(mag, ori, thresh=1.0)


# def block_based_ascii_representation(img, block_size=64, num_buckets=5):
#     height, width = img.shape
#     ascii_image = np.zeros_like(img)
    
#     # Divide the image into blocks of size block_size x block_size
#     for i in range(0, height, block_size):
#         for j in range(0, width, block_size):
#             block = img[i:i + block_size, j:j + block_size]
            
#             # Flatten and count the occurrence of each bucket
#             flat_block = block.flatten()
#             bucket_counts = np.histogram(flat_block, bins=num_buckets, range=(0, 255))[0]
            
#             # Elect the most frequent bucket as the representative
#             representative_bucket = np.argmax(bucket_counts)
            
#             # Map the representative bucket to an ASCII character index
#             ascii_char_index = representative_bucket * (255 // num_buckets)
            
#             # Assign the representative character to the entire block
#             ascii_image[i:i + block_size, j:j + block_size] = ascii_char_index

#     return ascii_image
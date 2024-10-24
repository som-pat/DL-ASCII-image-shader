import os
import cv2
import numpy as np
from iofile import *
from math import floor,ceil
import collections



def canny_edge_detection(image, low_threshold=50, high_threshold=220):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 1.4) 
    edges = cv2.Canny(blurred_image, low_threshold, high_threshold)
    # display(edges)
    return edges


def lab_contrast_enhance(img, factor):
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_img[:, :, 0] = cv2.multiply(lab_img[:, :, 0], factor) #1.182
    lab_img[:, :, 0] = clahe.apply(lab_img[:, :, 0])  
    final_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)

    ## display(final_img)
    return final_img




def desat_graysc(img,cond):

    if cond==0:
        grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ## display(grayscale_image)
        return grayscale_image
    
    elif cond ==1:
        b = img[:,:,0].astype(np.float32)
        g = img[:,:,1].astype(np.float32)
        r = img[:,:,2].astype(np.float32)       

        desat = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
        
        ## display(desat)
        return desat

    elif cond==2:
        b = img[:,:,0].astype(np.float32)
        g = img[:,:,1].astype(np.float32)
        r = img[:,:,2].astype(np.float32)
        desat = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
        grayscale_image = cv2.equalizeHist(desat)
        
        ## display(grayscale_image)
        return grayscale_image


def sharpen(img):
    kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel)
    return img

def fetch_ascii_char(ascii_image, char_index, ascii_len, char_size=(8, 8)):
    
    x = (char_index % ascii_len) * char_size[0]  # Horizontal position
    y = (char_index // ascii_len) * char_size[1]  # Vertical position    
    # Crop ASCII character 
    ascii_char = ascii_image[y:y + char_size[1], x:x + char_size[0]]
    
    return ascii_char

def luminance_to_ascii_index(luminance, num_buckets=10):    
    return  floor((luminance / 255) * (num_buckets - 1))


im_count = 0
def process_image_ascii(img):
    ascii_img = cv2.imread('ASCII_inside.png',cv2.IMREAD_GRAYSCALE) #grayscale for (8,80,3) to (8,80)
    char_size = (8,8)
    global im_count     
    ascii_len = 10
    dicte = {}

    ascii_art_image = np.zeros_like(img) #Empty board creation
    print(ascii_art_image.shape)

    for i in range(0, img.shape[0],char_size[0]):
        for j in range(0, img.shape[1],char_size[1]):   
            block = img[i:i + char_size[0], j:j + char_size[1]]
            
            if block.shape[0] != 8 or block.shape[1] != 8:
                continue
            
              
            average_luminance = np.mean(block)
            # theresholding whether to use np.max            

            ascii_index = luminance_to_ascii_index(average_luminance)
            # if ascii_index != 0:
            #     print(i,j)
            #     print(block)
            #     print(ascii_index)
            dicte[ascii_index] = dicte.get(ascii_index,0)+1  

            ascii_char = fetch_ascii_char(ascii_img, ascii_index, ascii_len, char_size)       
            
            ascii_art_image[i:i + char_size[0], j:j + char_size[1]] = ascii_char
            
    
    im_count +=1
    ## display(ascii_art_image)
    print(dicte)
    file_name = 'segment/ascii' + str(im_count) + '.jpg'
    cv2.imwrite(file_name, ascii_art_image)
    return ascii_art_image


count = 0 
img_loc = 'samples'
for file_name in os.listdir(img_loc):
    f = os.path.join(img_loc,file_name)
    if os.path.isfile(f):
        print()
        # if count == 2:
        #     break
        # count+=1

        img = cv2.imread(f)
        img = image_dimension(img)
        img = lab_contrast_enhance(img, factor=1.052)
        img = cv2.fastNlMeansDenoising(img, None, 20, 7, 21)
        image1 = desat_graysc(img, 2)
        image1 = canny_edge_detection(image1)
        img = lab_contrast_enhance(img, factor= 1.082)
        # img = sharpen(img) 
        img = desat_graysc(img,1)
        img = cv2.medianBlur(img,9)
        img = cv2.GaussianBlur(img,(7,7),0)
        img = up_down_scaling(img, block_size= 8)      
        img = process_image_ascii(img)

        img = cv2.addWeighted(img, 0.78, image1, 0.22, 0)
        # display(img)
        print(img.shape)
        count+=1
        file_name = 'segement_final/ascii' + str(count) + '.jpg'
        cv2.imwrite(file_name, img)
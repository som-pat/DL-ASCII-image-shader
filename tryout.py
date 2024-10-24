import cv2
import numpy as np
import os
from iofile import *
import matplotlib.pyplot as plt




    

def fetch_ascii_char(ascii_image, char_index, ascii_len, char_size=(8, 8)):
    
    x = (char_index % ascii_len) * char_size[0]  # Horizontal position
    y = (char_index // ascii_len) * char_size[1]  # Vertical position    
    # Crop the specific ASCII character 
    ascii_char = ascii_image[y:y + char_size[1], x:x + char_size[0]]
    return ascii_char

def luminance_to_ascii_index(luminance, num_buckets=10):    
    return int((luminance / 255) * (num_buckets - 1))

def angle_to_ascii_index(average_angle,num_buckets = 5):
    return int((average_angle/255)* (num_buckets -1))


im_count = 0
def process_image_ascii(img):
    ascii_img = cv2.imread('ASCII_inside.png',cv2.IMREAD_GRAYSCALE) #grayscale for (8,80,3) to (8,80)
    char_size = (8,8)
    global im_count     
    ascii_len = 10

    ascii_art_image = np.zeros_like(img) #Empty board creation
    print(ascii_art_image.shape)

    for i in range(0, img.shape[0],char_size[0]):
        for j in range(0, img.shape[1],char_size[1]):   
            block = img[i:i + char_size[0], j:j + char_size[1]]
            
            if block.shape[0] != 8 or block.shape[1] != 8:
                continue
            average_luminance = np.mean(block)

            ascii_index = luminance_to_ascii_index(average_luminance)  

            ascii_char = fetch_ascii_char(ascii_img, ascii_index, ascii_len, char_size)             
            
            ascii_art_image[i:i + char_size[0], j:j + char_size[1]] = ascii_char
            
    
    im_count +=1
    display(ascii_art_image)
    file_name = 'result/ascii' + str(im_count) + '.jpg'
    cv2.imwrite(file_name, ascii_art_image)
    return ascii_art_image
    

ed_count = 0
def edge_ascii_image(img):
    edge_ascii=cv2.imread('edgesASCII.png',cv2.IMREAD_GRAYSCALE)
    char_size = (8,8)
    global ed_count
    ascii_len = 5

    edge_art = np.zeros_like(img)
    dicte = {}
    for i in range(0,img.shape[0],char_size[0]):
        for j in range(0, img.shape[1],char_size[1]):

            edge = img[i:i+char_size[1], j:j+char_size[0]]
            if edge.shape[0] != 8 or edge.shape[1] != 8:
                continue

            average_angle = np.mean(edge)
            edge_index = angle_to_ascii_index(average_angle)
            dicte[edge_index] = dicte.get(edge_index,0)+1

            edge_char = fetch_ascii_char(edge_ascii, edge_index, ascii_len, char_size)
            edge_art[i:i+ char_size[0],j:j+char_size[1]] = edge_char
    print(dicte)
    display(edge_art)
    ed_count +=1
    file_name = 'result/saturation_edge_ascii_' + str(ed_count) + '.jpg'
    cv2.imwrite(file_name, edge_art)
    return edge_art
    


def dilation(img):
    kernel = np.ones((3,3),np.uint8)
    dilate_img = cv2.dilate(img, kernel, iterations=1)
    return dilate_img



count = 0 
img_loc = 'samples'
for file_name in os.listdir(img_loc):
    f = os.path.join(img_loc,file_name)
    if os.path.isfile(f):
        print()
        if count ==1:
            break
        count+=1
        
        ##Inner 2nd part 
        img = cv2.imread(f)
        img = image_dimension(img)
        img = image_sharpen(img)
        img = lab_contrast_enhance(img)
        img = desat_graysc(img,False)
        img = up_down_scaling(img, block_size= 8)        
        img = process_image_ascii(img)
        
        ## edge part
        edge = cv2.imread(f)
        edge = image_dimension(edge)
        edge = enhance_edges(edge,saturation=1.42, value=0.5, lightness=1.22)
        edge = cv2.fastNlMeansDenoising(edge, None, 20, 7, 21)
        edge = difference_of_Gaussian(edge,kernel1=17,kernel2=13,sigma1=0,sigma2=15)
        # edge = extended_sobel(edge)
        edge = gradient_direction(edge)        
        edge = desat_graysc(edge,False)
        edge = up_down_scaling(edge, block_size= 8)  
        edge = edge_ascii_image(edge)

        combi = overlay_images(img,edge)
        # rcombi = red_shader(combi)
       





        



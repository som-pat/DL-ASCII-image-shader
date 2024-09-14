import cv2
import numpy as np
import os
from iofile import *



def image_dimension(img):
    if img.shape[0]>512 and img.shape[1]>512:
        img = cv2.resize(img, (1024,1024), interpolation=cv2.INTER_AREA)
        display(img)
    else:
        img = cv2.resize(img, (512,512), interpolation=cv2.INTER_AREA)
        display(img)
    
    return img


def up_down_scaling(img, block_size):

    down_scaling = cv2.resize(img,(img.shape[1] // block_size, img.shape[0] // block_size), interpolation=cv2.INTER_AREA)    

    up_scaling = cv2.resize(down_scaling, (down_scaling.shape[1] * block_size, down_scaling.shape[0] * block_size), 
                            interpolation=cv2.INTER_NEAREST)
    display(up_scaling)

    return up_scaling

def desat_graysc(img):
    grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print('grayscale_image')
    

    desaturated_image = np.mean(img, axis=2).astype(np.uint8)
    print('desaturated_image')
    display(desaturated_image)

    return desaturated_image

def fetch_ascii_char(ascii_image, char_index, ascii_len, char_size=(8, 8)):
    
    x = (char_index % ascii_len) * char_size[0]  # Horizontal position
    y = (char_index // ascii_len) * char_size[1]  # Vertical position
    
    # Crop the specific ASCII character 
    ascii_char = ascii_image[y:y + char_size[1], x:x + char_size[0]]
    return ascii_char

def luminance_to_ascii_index(luminance, num_buckets=10):
    # Convert luminance to a bucket index (0 to 9)
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

ed_count = 0
def edge_ascii_image(img):
    edge_ascii=cv2.imread('edgesASCII.png',cv2.IMREAD_GRAYSCALE)
    char_size = (8,8)
    global ed_count
    ascii_len = 5

    edge_art = np.zeros_like(img)
    print(edge_art.shape)

    for i in range(0,img.shape[0],char_size[0]):
        for j in range(0, img.shape[1],char_size[1]):

            edge = img[i:i+char_size[1], j:j+char_size[0]]
            if edge.shape[0] != 8 or edge.shape[1] != 8:
                continue

            average_angle = np.mean(edge)
            edge_index = angle_to_ascii_index(average_angle)
            edge_char = fetch_ascii_char(edge_ascii, edge_index, ascii_len, char_size)
            edge_art[i:i+ char_size[0],j:j+char_size[1]] = edge_char
    
    display(edge_art)
    ed_count +=1
    file_name = 'result/edge_ascii_' + str(ed_count) + '.jpg'
    cv2.imwrite(file_name, edge_art)


img_loc = 'samples'
for file_name in os.listdir(img_loc):
    f = os.path.join(img_loc,file_name)
    if os.path.isfile(f):
        #1st part
        img = cv2.imread(f)
        img = image_dimension(img)
        img = image_sharpen(img)
        img = desat_graysc(img)
        img = enhance_contrast(img)
        img = difference_of_Gaussian(img,sigma1=0.1,sigma2=6.5)        
        img = extended_sobel(img)
        res_up = up_down_scaling(img, block_size= 8)
        # res_up = desat_graysc(res_up)
        print('res_up',res_up.shape)
        process_image_ascii(res_up)

        #2nd part
        img2 = cv2.imread(f)
        img2 = image_dimension(img2)
        img2 = desat_graysc(img2)
        img2 = cv2.fastNlMeansDenoising(img2, None, 20, 7, 21) 
        print('Noise')
        display(img2)
        img2 = image_sharpen(img2)
        print('contrast-sharpen')
        display(img2)
        img2 = enhance_contrast(img2)
        print('contrast')
        display(img2)
        # img2 = cv2.fastNlMeansDenoising(img2, None, 20, 7, 21) 
        # print('Noise')
        # display(img2)
        img2 = difference_of_Gaussian(img2,sigma1=2.5,sigma2=6.0)
        print('contrast-dog')
        display(img2)

        img2 = gradient_direction(img2)
        display(img2)

        img2 = up_down_scaling(img2,block_size=8)
        edge_ascii_image(img2)



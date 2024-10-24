import os
import cv2
import numpy as np
from iofile import *
from math import floor,ceil
import collections




dog_count = 0
def dog(img, kernel1, kernel2, sigma1, sigma2, tau, th):
    
    k = 1.6
    sigma2 = sigma2 *k
    p = 20.7

    grad1 = cv2.GaussianBlur(img,(kernel1,kernel1),sigma1) #edge 15
    grad2 = cv2.GaussianBlur(img,(kernel2, kernel2),sigma2) #edge 13
    

    dog_sigma_k = ((1+tau) * grad1) - (tau * grad2)
    dog_sigma_k = grad1 + p * dog_sigma_k


    #np.minimum(2*dog_sigma_k, 255)
    

    if th < np.percentile(dog_sigma_k, 97):
        th = np.percentile(dog_sigma_k, 97)
        print(th)

    dog_img = np.where(dog_sigma_k > th, np.minimum(2*dog_sigma_k, 255), 0 ).astype(np.uint8)
    print('Min:',np.min(dog_img),'Max:',np.max(dog_img))

    #display(dog_img)
    global dog_count
    dog_count +=1
    file_name = 'dog/dog_' + str(dog_count) + '.jpg'
    cv2.imwrite(file_name, dog_img) 
    return dog_img



def block_histogram(img_block, block_thresh):
    flat_block = img_block.flatten()
    block_counter = collections.Counter(flat_block)   
    
    non_zero_count = sum(v for k, v in block_counter.items() if k != 0)
    if non_zero_count < block_thresh:
        return 0    
    
    if 0 in block_counter:
        del block_counter[0.0]    
    
    angle_sum = sum(k * v for k, v in block_counter.items())
    angle_average = round(angle_sum / non_zero_count)
    img_block = np.where(img_block > 0 , angle_average, 0)
    return img_block



def sobel_filter(img):
    Sx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    Sy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    grad_theta = np.arctan2(Sy, Sx)
    magnitude = np.sqrt(Sx **2 + Sy**2)

    return grad_theta, magnitude

oric = 0
def gradient_direction(edge, edge_Threshold, block_threshold = 10):
    grad_theta, magnitude = sobel_filter(edge)
    
    mag_edges = np.where(magnitude >= edge_Threshold, 255, 0).astype(np.uint8)
    
    grad_theta = np.degrees(grad_theta)
    print('Macx', np.max(grad_theta), 'Minx', np.min(grad_theta))
    new_grad = np.where(grad_theta < 0, ((180 - (grad_theta))%180) , grad_theta)
    new_grad = np.round(new_grad)
    # grad_theta = ((grad_theta / 180) + 1) *90
    # print('Macx', np.max(grad_theta), 'Minx', np.min(grad_theta))

    # grad_theta = ((grad_theta+ np.pi)/(np.pi))

    grad_theta = np.round(grad_theta)
   

    char_size = (8,8)
    orient_map = np.zeros_like(edge)
    for i in range(0, mag_edges.shape[0], char_size[0]):
        for j in range(0, mag_edges.shape[1], char_size[1]):
            
            mag_block = mag_edges[i:i + char_size[0], j:j + char_size[1]]
            grad_block = grad_theta[i:i + char_size[0], j:j + char_size[1]]
            new_grad_block = new_grad[i:i + char_size[0], j:j + char_size[1]]
            
            
            if np.max(mag_block) !=0:
                # print(i,j)
                # print(mag_block)
                # print(grad_block)
                # print()
                # print(new_grad_block)
                # print()                
                
                array_max_block = block_histogram(new_grad_block,block_threshold)
                
                # print(array_max_block)
                # print()
                # print('-'*20)
                orient_map[i:i + char_size[0], j:j + char_size[1]] = array_max_block
                
            else:
                orient_map[i:i + char_size[0], j:j + char_size[1]] = new_grad_block
            

    print('orent')
    # display(orient_map)
    # print(orient_map)
    orient_map = cv2.medianBlur(orient_map,3)
    counter = collections.Counter(orient_map.flatten())
    print( counter.most_common())
    global oric 
    oric +=1
    file_name = 'orient/orient_' + str(oric) + '.jpg'
    cv2.imwrite(file_name, orient_map) 
    

    return orient_map












    
    

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
            dicte[ascii_index] = dicte.get(ascii_index,0)+1  

            ascii_char = fetch_ascii_char(ascii_img, ascii_index, ascii_len, char_size)       
            
            ascii_art_image[i:i + char_size[0], j:j + char_size[1]] = ascii_char
            
    
    im_count +=1
    #display(ascii_art_image)
    print(dicte)
    file_name = 'segment/ascii' + str(im_count) + '.jpg'
    cv2.imwrite(file_name, ascii_art_image)
    return ascii_art_image



def edge_char_mapping(edge_ascii, edge_index, ascii_len ,char_size):
    x = edge_index  * char_size[0] # 0-40
    y = 0
    ascii_char = edge_ascii[y:y + char_size[1], x:x + char_size[0]]
    return ascii_char

def angle_to_ascii_index(angle):
    if angle == 0: return 0
    elif (angle>0 and angle <=23) or (angle>=157 and angle<=180): return 1
    elif (angle>=24 and angle <=78): return 4
    elif (angle>=79 and angle <=104): return 2
    elif (angle>=105 and angle<=156): return 3 


ed_count = 0
def ascii_edge_mapping(edge):
    edge_ascii=cv2.imread('edgesASCII.png',cv2.IMREAD_GRAYSCALE)
    char_size = (8,8)
    global ed_count
    ascii_len = 5

    edge_map = np.zeros_like(edge)
    dicte = {}
    for i in range(0,edge.shape[0],char_size[0]):
        for j in range(0, edge.shape[1],char_size[1]):

            edge_block = edge[i:i+char_size[1], j:j+char_size[0]]
            average_angle = np.max(edge_block)
            
            edge_index = angle_to_ascii_index(average_angle)
            if edge_index is None:
                print('edge_index',edge_index)
            dicte[edge_index] = dicte.get(edge_index,0)+1
            edge_char = edge_char_mapping(edge_ascii,edge_index, ascii_len ,char_size)
            edge_map[i:i+char_size[0],j:j+char_size[1]] = edge_char

    # display(edge_map)
    print(dicte)
    ed_count +=1
    file_name = 'result/saturation_edge_ascii_' + str(ed_count) + '.jpg'
    cv2.imwrite(file_name, edge_map)
    return edge_map


count = 0 
img_loc = 'samples'
for file_name in os.listdir(img_loc):
    f = os.path.join(img_loc,file_name)
    if os.path.isfile(f):
        print()
        # if count == 6:
        #     break
        # count+=1
        
        ##Inner 2nd part 
        img = cv2.imread(f)
        img = image_dimension(img)
        image1 = desat_graysc(img, 2)
        img = lab_contrast_enhance(img)
        img = sharpen(img) 
        img = desat_graysc(img,1)
        # img = cv2.fastNlMeansDenoising(img, None, 20, 7, 21)
        # img = cv2.medianBlur(img,7)
        # img = cv2.GaussianBlur(img,(7,7),0)
        img = up_down_scaling(img, block_size= 8)      
        img = process_image_ascii(img)

        img = cv2.addWeighted(img, 0.5, image1, 0.2, 0)



        edge = cv2.imread(f)
        edge = image_dimension(edge)
        edge = enhance_edges(edge,saturation=1.32, value=0.85, lightness=1.29)
        edge = desat_graysc(edge,2) 
        edge = cv2.fastNlMeansDenoising(edge, None, 20, 7, 21)
        edge = image_sharpen(edge)
        edge = cv2.medianBlur(edge,3)
        edge = dog(edge,kernel1=13,kernel2=17,sigma1=1, sigma2 = 4.2,tau=0.915, th=80)
        print('dog')
        # edge = cv2.medianBlur(edge,1)        
        edge = gradient_direction(edge, edge_Threshold = 42)
        # edge = up_down_scaling(edge, block_size=2)
        edge = ascii_edge_mapping(edge)
        
        print('-'*10)

        combi = overlay_images(img,edge)
        
        
 
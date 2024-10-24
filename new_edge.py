import os
import cv2
import numpy as np
from iofile import *
from math import floor,ceil
import collections
from skimage.morphology import skeletonize

dog_count = 0

# def horizontal_blur(image, kernel_size, sigma1, sigma2):
#     return cv2.GaussianBlur(image, (kernel_size, 1), sigma1), cv2.GaussianBlur(image, (kernel_size, 1), sigma2)

# def vertical_blur(image, kernel_size, sigma1, sigma2):
#     # Apply vertical Gaussian blur
#     return cv2.GaussianBlur(image, (1, kernel_size), sigma1), cv2.GaussianBlur(image, (1, kernel_size), sigma2)

# def new_dog(image, kernel1,kernel2, sigma1, tau, th):
#     k = 1.6
#     sigma2 = sigma1 * k 
#     blur_x, blur_x_scaled = horizontal_blur(image, kernel2, sigma1, sigma2)
#     blur_y, blur_y_scaled = vertical_blur(image, kernel2, sigma1, sigma2)
    
#     # Calculate DoG
#     D = ((blur_x - tau * blur_x_scaled) + (blur_y - tau * blur_y_scaled)) / 2.0
    
#     # Apply threshold
#     dog_img = np.where(D >= th, 255, 0).astype(np.uint8)
#     global dog_count
#     dog_count +=1
#     file_name = 'dog/dog_' + str(dog_count) + '.jpg'
#     cv2.imwrite(file_name, dog_img) 
#     return dog_img
# can_count=0
# def canny_edge_detection(image, low_threshold=50, high_threshold=240):
#     blurred_image = cv2.GaussianBlur(image, (5, 5),1.4) 
#     edges = cv2.Canny(blurred_image, low_threshold, high_threshold)
#     #display(edges)
    
#     global can_count
#     can_count +=1
#     file_name = 'canny_edge/can_' + str(can_count) + '.jpg'
#     cv2.imwrite(file_name, edges) 
#     return edges    
    


def dog(img, kernel1, kernel2, sigma1, tau, th):    
    k = 1.6
    sigma2 = sigma1 * k 
    p = 11.5

    grad1 = cv2.GaussianBlur(img,(kernel1, kernel1),sigma1) 
    grad2 = cv2.GaussianBlur(img,(kernel2 ,kernel2),sigma2) 
    

    dog_sigma_k = ((1+tau) * grad1) - (tau * grad2)
    dog_sigma_k = grad1 + p * dog_sigma_k

    if th < np.percentile(dog_sigma_k, 85):
        th = np.percentile(dog_sigma_k, 85)
        # print(th)

    # np.minimum(2*dog_sigma_k, 255)
    dog_img = np.where(dog_sigma_k > th, dog_sigma_k, 0 ).astype(np.uint8)    

    #display(dog_img)
    global dog_count
    dog_count +=1
    file_name = 'dog/dog_' + str(dog_count) + '.jpg'
    cv2.imwrite(file_name, dog_img) 
    return dog_img


    


def block_histogram(img_block, block_thresh):
    flat_block = img_block.flatten()
    non_zero_values = flat_block[flat_block > 0]

    if len(non_zero_values) < block_thresh:
        return 0

    median_angle = int(np.mean(non_zero_values))
    img_block = np.where(img_block > 0, median_angle, 0)    
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
    grad_theta = np.round(grad_theta)
    ## print('Macx', np.max(grad_theta), 'Minx', np.min(grad_theta))
    new_grad = np.where(grad_theta < 0, ((180 - (grad_theta))%180) , grad_theta)
    new_grad = np.round(new_grad)

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
                new_grad_block = np.where(mag_block == 0, 0, new_grad_block)
                # print(new_grad_block)
                array_max_block = block_histogram(new_grad_block,block_threshold)
                
                # print(array_max_block)
                # print()
                # print('-'*20)
                orient_map[i:i + char_size[0], j:j + char_size[1]] = array_max_block
                
            else:
                orient_map[i:i + char_size[0], j:j + char_size[1]] = new_grad_block
            

    ## print('orent')
    # #display(orient_map)
    # ## print(orient_map)
    # orient_map = cv2.medianBlur(orient_map,3)
    counter = collections.Counter(orient_map.flatten())
    ## print( counter.most_common())
    global oric 
    oric +=1
    file_name = 'orient/orient_' + str(oric) + '.jpg'
    cv2.imwrite(file_name, orient_map)    

    return orient_map


def thin_edges(img):
    # Convert image to binary
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    binary_bool = binary_img > 0
    skeleton = skeletonize(binary_bool)
    thinned_img = (skeleton * 255).astype(np.uint8)
    #display(thinned_img)

    return thinned_img



def edge_char_mapping(edge_ascii, edge_index, ascii_len ,char_size):
    x = (edge_index % ascii_len)  * char_size[0] # 0-40
    y = 0
    ascii_char = edge_ascii[y:y + char_size[1], x:x + char_size[0]]
    return ascii_char

def angle_to_ascii_index(angle):
    if angle == 0: return 0
    elif (angle>0 and angle <=23) or (angle>=157 and angle<=180): return 1
    elif (angle>=24 and angle <=78): return 4
    elif (angle>=79 and angle <=108): return 2
    elif (angle>=109 and angle<=156): return 3 


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
            # if np.max(edge_block)>0:
            #     # print(12,edge_block)
            #     # print(23,average_angle)
            #     # print(34,edge_char)
            edge_map[i:i+char_size[0],j:j+char_size[1]] = edge_char

    # #display(edge_map)
    ## print(dicte)
    ed_count +=1
    file_name = 'edges/saturation_edge_ascii_' + str(ed_count) + '.jpg'
    cv2.imwrite(file_name, edge_map)
    return edge_map




count = 0 
img_loc = 'samples'
for file_name in os.listdir(img_loc):
    f = os.path.join(img_loc,file_name)
    if os.path.isfile(f):
        # print()
        # if count ==1:
        #     break
        # count+=1

        edge = cv2.imread(f)
        edge = image_dimension(edge)
        edge = enhance_edges(edge,saturation=1.32, value=0.85, lightness=1.29)
        
        edge = desat_graysc(edge,4)
        # edge = cv2.fastNlMeansDenoising(edge, None, 20, 7, 21)
        edge = cv2.medianBlur(edge,9)
        edge = image_sharpen(edge)
        # edge = canny_edge_detection(edge)
        edge = cv2.medianBlur(edge,3)
        
        edge = dog(edge, kernel1=3, kernel2 = 21, sigma1 = 0.98, tau=0.916, th =80)
        edge = cv2.medianBlur(edge,3)
        edge = gradient_direction(edge, edge_Threshold = 50, block_threshold = 10)
        edge = ascii_edge_mapping(edge)



        # edge = enhance_edges(edge,saturation=1.32, value=0.75, lightness=1.29)
        # edge = desat_graysc(edge,0) 
        # # edge = cv2.fastNlMeansDenoising(edge, None, 20, 7, 21)
        # # edge = image_sharpen(edge)
        # edge = canny_edge_detection(edge)
        # # edge = cv2.medianBlur(edge,3)
        # edge = dog(edge,kernel1=13,kernel2=17,sigma1=1.4, sigma2 = 4.2,tau=0.915, th=80)
        # # ## print('dog')
        # edge = thin_edges(edge)       
        # edge = gradient_direction(edge, edge_Threshold = 42)
        # edge = cv2.medianBlur(edge,5) 
        # # edge = up_down_scaling(edge, block_size=8)
        # edge = ascii_edge_mapping(edge)
        
        # # print('-'*50)
import cv2
import numpy as np
import math
from collections import defaultdict


def display(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def desat_graysc(img,cond):

    if cond==0:
        grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #display(grayscale_image)
        return grayscale_image
    
    elif cond ==1:
        b = img[:,:,0].astype(np.float32)
        g = img[:,:,1].astype(np.float32)
        r = img[:,:,2].astype(np.float32)       

        desat = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
        
        #display(desat)
        return desat

    elif cond==2:
        b = img[:,:,0].astype(np.float32)
        g = img[:,:,1].astype(np.float32)
        r = img[:,:,2].astype(np.float32)
        desat = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
        grayscale_image = cv2.equalizeHist(desat)
        
        #display(grayscale_image)
        return grayscale_image
    
    elif cond==3:
        b = img[:,:,0].astype(np.float32)
        g = img[:,:,1].astype(np.float32)
        r = img[:,:,2].astype(np.float32)

        desat = (0.2126 * r + 0.7152 *g + 0.0722 *b).astype(np.uint8)
        return desat
    
    elif cond == 4:
        b = img[:,:,0].astype(np.float32)
        g = img[:,:,1].astype(np.float32)
        r = img[:,:,2].astype(np.float32)

        desat = np.sqrt( 0.299*r **2 + 0.587*g**2 + 0.114*b**2 ).astype(np.uint8)
        return desat

def up_down_scaling(img, block_size):

    down_scaling = cv2.resize(img,(img.shape[1] // block_size, img.shape[0] // block_size), interpolation=cv2.INTER_AREA)    

    up_scaling = cv2.resize(down_scaling, (down_scaling.shape[1] * block_size, down_scaling.shape[0] * block_size), 
                            interpolation=cv2.INTER_NEAREST)
    #display(up_scaling)

    return up_scaling



def image_dimension(img):
    if img.shape[0]>512 and img.shape[1]>512:
        img = cv2.resize(img, (1024,1024), interpolation=cv2.INTER_AREA)
        #display(img)
    else:
        img = cv2.resize(img, (512,512), interpolation=cv2.INTER_AREA)
        #display(img)
    
    return img



def hsv_val(img):
    hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV_FULL)
    h, s, val_img = cv2.split(hsv_img)
    val_img = cv2.multiply(val_img, 1.5)
    #display(val_img)
    val_img =  np.clip(val_img, 0, 255).astype(np.uint8)
    #display(val_img)
    filter_val_img = cv2.merge([h, s, val_img])
    #display(filter_val_img)
    filter_val_img = cv2.cvtColor( filter_val_img, cv2.COLOR_HSV2BGR)
    #display(filter_val_img)

    return filter_val_img



def threshold_saturation(img):
    hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV_FULL)
    hsv_img[:, :, 1] = cv2.multiply(hsv_img[:, :, 1], 1.55) 
    saturate_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR_FULL)    
    #display(saturate_img)

    return saturate_img

def satval_gradient(img):
    hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV_FULL)
    h, saturate_img, val_img = cv2.split(hsv_img)

    val_img = cv2.multiply(val_img, 1.1)
    val_img =  np.clip(val_img, 0, 255).astype(np.uint8)

    saturate_img = cv2.multiply(saturate_img, 1.5 )
    saturate_img = np.clip(saturate_img, 0, 255).astype(np.uint8)

    filter_img = cv2.merge([h, saturate_img, val_img])
    filter_img = cv2.cvtColor( filter_img, cv2.COLOR_HSV2BGR)
    
    #display(filter_img)
    return filter_img

def enhance_edges(image,saturation, value, lightness):
    # increase saturation for bright edge 
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    hsv_img[:, :, 1] = cv2.multiply(hsv_img[:, :, 1], saturation)  
    hsv_img[:, :, 2] = cv2.multiply(hsv_img[:, :, 2], value)  
    enhanced_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR_FULL)
    #display(enhanced_img)
    
    # Apply CLAHE for enhance local contrast
    lab_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_img[:, :, 0] = cv2.multiply(lab_img[:, :, 0], lightness)
    lab_img[:, :, 0] = clahe.apply(lab_img[:, :, 0])  
    final_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)

    #display(final_img)

    return final_img


def sharpen(img):
    kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel)
    return img



def image_sharpen(img): #Image sharpening kernel
    kernel = np.array([[0,-1, 0], [-1,4,-1], [0,-1,0]]) 
    img = cv2.filter2D(img, -1, kernel)
    #display(img)
    return img


def prewitt_filter(img):
    img = cv2.GaussianBlur(img,(3,3),0)
    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])

    px = cv2.filter2D(img, -1, kernelx)
    py = cv2.filter2D (img , -1, kernely)

    fin_img = cv2.addWeighted(px, 0.5, py, 0.5, 0)

    #display(fin_img)
    #display(px+py)
    return fin_img




def blurring(img):
    kernel = np.array([[0.0625,0.125,0.0625],[0.125,0.125,0.125],[0.0625,0.125,0.0625]])
    img = cv2.filter2D(img, -1, kernel)
    #display(img)
    return img





def gradient_direction(img):
    img = cv2.GaussianBlur(img,(5,5),0)
    Sx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    Sy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    # Sx = np.mean(Sx)
    # Sy = np.mean(Sy)


    grad_theta = np.atan2(Sy,Sx)/(2*np.pi)
    print('Min:',np.min(grad_theta))
    print('Max:',np.max(grad_theta))

         
    theta_normalise = np.add(grad_theta,0.5,where = grad_theta!=0)    
    theta_image = (theta_normalise*255).astype(np.uint8)

    theta_image = np.floor(theta_image)
    return theta_image

def new_gradient_orient(img):
    img = cv2.GaussianBlur(img,(5,5),0)
    Sx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    Sy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3) 

    orient = np.arctan2(Sy, Sx) #[-pi , pi] 
    orient = np.where(orient < 0, orient + 2 * np.pi, orient)
    pixel_value = (orient / (2 * np.pi)) * 255
  
    #display(pixel_value)
    return pixel_value


def extended_sobel(img):
    grad_x = cv2.Sobel(img, cv2.CV_16S, 1, 0,ksize=5,scale=1,delta=0,borderType=cv2.BORDER_DEFAULT) #(x:1,y:0) 16bit gradient change along X axis
    grad_y = cv2.Sobel(img, cv2.CV_16S, 0, 1,ksize=5,scale=1,delta=0,borderType=cv2.BORDER_DEFAULT) #(x:0,y:1) 16bit gradient change along y axis
    
    abs_grad_x = cv2.convertScaleAbs(grad_x) #Scales, calculates absolute values, and converts the result to 8-bit.
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0) #Calculates the weighted sum of two arrays.    
    output = grad.astype(np.uint8)
    print('sobel')
    
    return output


def difference_of_Gaussian(img, kernel1, kernel2, sigma1, sigma2):
    grad1 = cv2.GaussianBlur(img,(kernel1,kernel1),sigma1) #edge 15
    grad2 = cv2.GaussianBlur(img,(kernel2, kernel2),sigma2) #edge 13
    DoG_img = cv2.subtract(grad1, grad2)
    
    #display(DoG_img) 
    return DoG_img


def lab_contrast_enhance(img):
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_img[:, :, 0] = cv2.multiply(lab_img[:, :, 0], 1.182)
    lab_img[:, :, 0] = clahe.apply(lab_img[:, :, 0])  
    final_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)

    #display(final_img)
    return final_img



def enhance_contrast(img):
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    en_img = clahe.apply(img)
    #display(en_img)
    return en_img







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
    
    #display(ori_map)
    return ori_map


def sobel_grad_orient(img):
    Sx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    Sy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    # Calculate magnitude and orientation of gradients
    mag = cv2.magnitude(Sx, Sy)
    #display(mag)
    ori = cv2.phase(Sx, Sy, angleInDegrees=True)
    #display(ori)
    ori_map = orientation_map(mag, ori, thresh=1.0)


def local_contrast_normalization(image, kernel_size=15, epsilon=1e-5):
    image = image.astype(np.float64)
    
    local_mean = cv2.blur(image, (kernel_size, kernel_size))
    squared_diff = (image - local_mean) ** 2
    local_variance = cv2.blur(squared_diff, (kernel_size, kernel_size))
    local_stddev = np.sqrt(local_variance + epsilon)
    
    lcn_image = (image - local_mean) / local_stddev
    lcn_image = cv2.normalize(lcn_image, None, 0, 255, cv2.NORM_MINMAX)
    lcn_image = lcn_image.astype(np.uint8)
    #display(lcn_image)
    
    return lcn_image

com_count = 0

def overlay_images(img1, img2):    
    
    _, mask = cv2.threshold(img2, 50, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    edges_from_img2 = cv2.bitwise_and(img2, img2, mask=mask)
    filled_from_img1 = cv2.bitwise_and(img1, img1, mask=mask_inv)
    # combined_image = cv2.add(edges_from_img2, filled_from_img1)
    combined_image = cv2.addWeighted(edges_from_img2, 0.85, filled_from_img1, 1, 0)
    # combined_image = cv2.fastNlMeansDenoising(combined_image, h=10, templateWindowSize=7, searchWindowSize=21)

    #display(combined_image)
    global com_count
    com_count +=1
    file_loc = 'result/combined_ascii'+str(com_count)+'.jpg'
    cv2.imwrite(file_loc, combined_image)
    return combined_image



red_count = 0

def mix_color_shader(ascii_img, original, gamma=1.2):

    ascii_norm = cv2.normalize(ascii_img.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)

    b, g, r = cv2.split(original)

    b = cv2.multiply(b.astype(np.float32), ascii_norm)
    g = cv2.multiply(g.astype(np.float32), ascii_norm)
    r = cv2.multiply(r.astype(np.float32), ascii_norm)

    colored_ascii = cv2.merge([b, g, r]).astype(np.uint8)
    display(colored_ascii)

    inv_gamma = 1.0 / gamma
    lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    cac= cv2.LUT(colored_ascii, lut)
    
    global red_count 
    red_count += 1
    file_loc = 'color_shade/shade'+str(red_count)+'.jpg'
    cv2.imwrite(file_loc, cac) 


    return cac

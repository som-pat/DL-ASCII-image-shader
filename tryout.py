import cv2
import numpy as np


def display(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def image_dimension(img):
    if img.shape[0]>512 and img.shape[1]>512:
        img = cv2.resize(img, (1024,1024), interpolation=cv2.INTER_AREA)
        display(img)
    else:
        img = cv2.resize(img, (512,512), interpolation=cv2.INTER_AREA)
        display(img)
    
    return img

def image_filter(img): #Image sharpening kernel
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) / 1
    img = cv2.filter2D(img, cv2.CV_8U, kernel)
    display(img)

    return img

def up_down_scaling(img, block_size):

    down_scaling = cv2.resize(img,(img.shape[1] // block_size, img.shape[0] // block_size), interpolation=cv2.INTER_AREA)    

    up_scaling = cv2.resize(down_scaling, (down_scaling.shape[1] * block_size, down_scaling.shape[0] * block_size), 
                            interpolation=cv2.INTER_NEAREST)
    display(up_scaling)

    return up_scaling, down_scaling

def desat_graysc(img):
    grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print('grayscale_image')
    

    desaturated_image = np.mean(img, axis=2).astype(np.uint8)
    print('desaturated_image')
    display(desaturated_image)

    return desaturated_image

def fetch_ascii_char(ascii_image, char_index, char_size=(8, 8)):
    x = (char_index % 10) * char_size[0]  # Horizontal position
    y = (char_index // 10) * char_size[1]  # Vertical position
    
    # Crop the specific ASCII character 
    ascii_char = ascii_image[y:y + char_size[1], x:x + char_size[0]]
    return ascii_char

def luminance_to_ascii_index(luminance, num_buckets=10):
    # Convert luminance to a bucket index (0 to 9)
    return int((luminance / 255) * (num_buckets - 1))

im_count = 0
def process_image_ascii(img):
    ascii_img = cv2.imread('ASCII_inside.png',cv2.IMREAD_GRAYSCALE)
    char_size = (8,8)
    global im_count     
    
    ascii_art_image = np.zeros_like(img)
    print(ascii_art_image.shape)

    for i in range(0, img.shape[0],char_size[0]):
        for j in range(0, img.shape[1],char_size[1]):   
            block = img[i:i + char_size[1], j:j + char_size[0]]
            
            if block.shape[0] != 8 or block.shape[1] != 8:
                continue
            average_luminance = np.mean(block)

            ascii_index = luminance_to_ascii_index(average_luminance)          

            ascii_char = fetch_ascii_char(ascii_img, ascii_index, char_size)   
             
            
            ascii_art_image[i:i + char_size[1], j:j + char_size[0]] = ascii_char
    im_count +=1
    display(ascii_art_image)
    file_name = 'result/ascii' + str(im_count) + '.jpg'
    cv2.imwrite(file_name, ascii_art_image)
    


img_loc = ['samples/bird.jpg','samples/sample.jpg','samples/bird2.jpg',
           'samples/bird3.jpg','samples/example6.png','samples/r_sample.jpg']
for img in img_loc:
    img = cv2.imread(img)
    img = image_dimension(img)
    img = image_filter(img)
    res_up, res_down = up_down_scaling(img, block_size= 8)
    res_up = desat_graysc(res_up)
    print('res_up',res_up.shape)
    process_image_ascii(res_up)
    
    
    
    # print(img.shape, res_up.shape, res_down.shape)
    # print(img.shape[0]/res_down.shape[0], img.shape[1]/res_down.shape[1])

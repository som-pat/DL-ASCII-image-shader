# import cv2
# import numpy as np
# import matplotlib.pyplot as plt 
# from PIL import Image, ImageFont, ImageDraw

# img_file = 'samples/bird sample.jpg'

# img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE) #, cv2.IMREAD_GRAYSCALE
# gray = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

# #possible replace for downscaling then upscaling
# # def subsample_image(image, pixel_size):
# #     # Subsampling the image by selecting pixels at intervals of 'pixel_size'
# #     subsampled = image[::pixel_size, ::pixel_size]
# #     display(subsampled)
# #     return subsampled

# # def upscale_image(image, scale_factor):
# #     # Upscale the image back to original size using nearest-neighbor interpolation
# #     upscaled = cv2.resize(image, 
# #                           (image.shape[1] * scale_factor, image.shape[0] * scale_factor), 
# #                           interpolation=cv2.INTER_NEAREST)
# #     display(upscaled)
# #     return upscaled


# def display(img):
#     cv2.imshow('image',img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# def down_up_scaling(image,down_factor):    
#     down = cv2.resize(image, (0,0), fx=1/down_factor, fy=1/down_factor, interpolation=cv2.INTER_LINEAR)

#     o_size = (image.shape[1], image.shape[0])
#     upscale = cv2.resize(down, o_size, interpolation=cv2.INTER_NEAREST)

#     return upscale

# def ascii_to_image(ascii_img, output_path='ascii_image.png'):    

#     # font check
#     font_path='arial.ttf' 
#     font_size=8

#     font = ImageFont.truetype(font_path, font_size)
#     bbox =  font.getbbox("A")
#     char_width, char_height = bbox[2] - bbox[0] , bbox[3] - bbox[1]  
#     print(font.getbbox("A")[:])
    
#     print('font',char_width, char_height)
#     img_width = char_width * len(ascii_img[0])
#     img_height = char_height * len(ascii_img)

#     print('Scaled image Width and height',img_width,img_height)

#     # Create a new blank image with white background
#     img = Image.new('RGB', (img_width, img_height), 'white')
#     draw = ImageDraw.Draw(img)

#     # Draw ASCII characters on the image
#     y = 0
#     for row in ascii_img:
#         draw.text((0, y), "".join(row), font=font, fill='black',spacing=0)
#         y+=char_height

#     img = img.rotate(-270, expand=True)#1568*1792
    
#     img.save(output_path)
#     img.show()


# def ascii_conversion(gray):
#     ascii_ren = []
#     ascii_characters = ["@", "#", "S", "%", "?", "*", "+", ";", ":", ",", ".", " "]

#     for i in range(gray.shape[0]):
#         row = []
#         for j in range(gray.shape[1]):
            
#             intensity  = gray[i,j]
#             intensity_index = int(intensity / (255 / (len(ascii_characters) - 1)))

#             try:
#                 ascii_ch = ascii_characters[intensity_index]
#             except:
#                 print(intensity_index)
#                 break
#             row.append(ascii_ch)

#         ascii_ren.append(row)    
#     ascii_to_image(ascii_ren)


# resdown_image = down_up_scaling(gray,8)
# ascii_conversion(resdown_image)

# display(img)
# display(resdown_image)




# # hist = cv2.calcHist([img], [0], None, [256], [0, 256])
# # hist = hist.reshape(256)

# # Plot histogram
# # plt.bar(np.linspace(0,255,256), hist)
# # plt.title('Histogram')
# # plt.ylabel('Frequency')
# # plt.xlabel('Grey Level')
# # plt.show()
# # cv2.imshow('Downscaled Image', upscale_image)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# img_loc = ['samples/bird.jpg','samples/sample.jpg','samples/bird2.jpg',
#            'samples/bird3.jpg','samples/example6.png','samples/r_sample.jpg']
# for img in img_loc:
    # img = cv2.imread(img)
    # img = image_dimension(img)
    # img = image_filter(img)
    # res_up, res_down = up_down_scaling(img, block_size= 8)
    # res_up = desat_graysc(res_up)
    # print('res_up',res_up.shape)
    # process_image_ascii(res_up)
    
    #dumpster
    # img2 = cv2.adaptiveThreshold(img2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
     #                                 cv2.THRESH_BINARY, 11, 2)
        # print('contrast-adthreshold')
        # display(img2)
    
    
    # print(img.shape, res_up.shape, res_down.shape)
    # print(img.shape[0]/res_down.shape[0], img.shape[1]/res_down.shape[1])


        # #2nd part
        # img2 = cv2.imread(f)
        # img2 = image_dimension(img2)
        # img2 = desat_graysc(img2,False)
        # # img2 = cv2.fastNlMeansDenoising(img2, None, 20, 7, 21) 
        # # print('Noise')
        # # display(img2)
        # img2 = image_sharpen(img2)
        # print('contrast-sharpen')
        # display(img2)
        # img2 = enhance_contrast(img2)
        # print('contrast')
        # display(img2)
        # img2 = cv2.fastNlMeansDenoising(img2, None, 20, 7, 21) 
        # print('Noise')
        # display(img2)
        # img2 = difference_of_Gaussian(img2,sigma1=3.9,sigma2=6.0)
        # print('contrast-dog')
        # display(img2)

        # img2 = extended_sobel(img2)
        # img2 = gradient_direction(img2)
        # display(img2)

        # # img2 = up_down_scaling(img2,block_size=8)
        # edge_ascii_image(img2)


        #3rd part 
        # img3 = cv2.imread(f)
        # img3 = image_dimension(img3)
        # # img3 = hsv_val(img3)
        # # img3 = threshold_saturation(img3)
        # # img3 = image_sharpen(img3)
        # # print('sharpen')
        # img3 = satval_gradient(img3)
        # img3 = desat_graysc(img3,False)

        # # img3 = up_down_scaling(img3,block_size=8)
        # # laplace = cv2.Laplacian(img3, cv2.CV_8U)
        # # display(laplace)
        # # img3 = cv2.normalize( img3, 0, 600, cv2.NORM_MINMAX)
        # # display(img3)
        # # img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
        # # display(img3)
        
        # # img3 = image_sharpen(img3)
        # # img3 = enhance_contrast(img3)
        # # print('enhance_contrast')
        # # display(img3)
        # img3 = image_sharpen(img3)
        # img3 = difference_of_Gaussian(img3,sigma1=2.0,sigma2=6.0)
        # print('difference_of_Gaussian')
        # display(img3)
        # img3 = gradient_direction(img3)
        # print('extended_sobel')
        # display(img3)
        # img3 = edge_ascii_image(img3) 

# def block_based_ascii_representation(img, block_size=64, num_buckets=5):
#     height, width = img.shape
#     ascii_image = np.zeros_like(img)
#     edge_ascii=cv2.imread('edgesASCII.png',cv2.IMREAD_GRAYSCALE)
    
#     # Divide the image into blocks of size block_size x block_size
#     for i in range(0, height, block_size):
#         for j in range(0, width, block_size):
#             block = img[i:i + block_size, j:j + block_size]
#             # print(1,block.shape)
#             # display(block)
            
            
#             # Flatten and count the occurrence of each bucket
#             flat_block = block.flatten()
#             print(2,flat_block, flat_block.shape)
#             bucket_counts = np.histogram(flat_block, bins=num_buckets, range=(0, 255))[0]
#             # print(bucket_counts, bucket_counts.shape)
#             # plt.figure(figsize=(8, 4))
#             # plt.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), edgecolor='black', align='edge')
#             # plt.title('Histogram of Pixel Intensities in Block')
#             # plt.xlabel('Pixel Intensity')
#             # plt.ylabel('Frequency')
#             # plt.xticks(bin_edges)  # Show bin edges on the x-axis
#             # plt.grid(axis='y')
#             # plt.show()
            
#             # Elect the most frequent bucket as the representative
#             representative_bucket = np.argmax(bucket_counts)
            
#             # Map the representative bucket to an ASCII character index
#             ascii_char_index = representative_bucket * (255 // num_buckets)
            
#             # Assign the representative character to the entire block
#             ascii_image[i:i + block_size, j:j + block_size] = ascii_char_index
    
#     display(ascii_image)
#     return ascii_image


# def block_based_ascii_representation(img, block_size=64, num_buckets=5):
#     height, width = img.shape
#     ascii_image = np.zeros_like(img)
#     edge_ascii=cv2.imread('edgesASCII.png',cv2.IMREAD_GRAYSCALE)
    
#     # Divide the image into blocks of size block_size x block_size
#     for i in range(0, height, block_size):
#         for j in range(0, width, block_size):
#             block = img[i:i + block_size, j:j + block_size]
#             # print(1,block.shape)
#             # display(block)
            
            
#             # Flatten and count the occurrence of each bucket
#             flat_block = block.flatten()
#             print(2,flat_block, flat_block.shape)
#             bucket_counts = np.histogram(flat_block, bins=num_buckets, range=(0, 255))[0]
#             # print(bucket_counts, bucket_counts.shape)
#             # plt.figure(figsize=(8, 4))
#             # plt.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), edgecolor='black', align='edge')
#             # plt.title('Histogram of Pixel Intensities in Block')
#             # plt.xlabel('Pixel Intensity')
#             # plt.ylabel('Frequency')
#             # plt.xticks(bin_edges)  # Show bin edges on the x-axis
#             # plt.grid(axis='y')
#             # plt.show()
            
#             # Elect the most frequent bucket as the representative
#             representative_bucket = np.argmax(bucket_counts)
            
#             # Map the representative bucket to an ASCII character index
#             ascii_char_index = representative_bucket * (255 // num_buckets)
            
#             # Assign the representative character to the entire block
#             ascii_image[i:i + block_size, j:j + block_size] = ascii_char_index
    
#     display(ascii_image)
#     return ascii_image



#2nd part (add 2nds to 4th and delete below)
        # img2 = cv2.imread(f)
        # img2 = image_dimension(img2)
        # # img2 = desat_graysc(img2,False)
        # # img2 = cv2.fastNlMeansDenoising(img2, None, 20, 7, 21) 
        # # print('Noise')
        # # display(img2)
        # # img2 = image_sharpen(img2)
        # # print('contrast-sharpen')
        # # display(img2)
        # img2 = enhance_edges(img2,saturation=1.32, value=0.8, lightness=1.22)
        # img2 = cv2.fastNlMeansDenoising(img2, None, 20, 7, 21) 
        # print('Noise')
        # display(img2)
        # img2 = difference_of_Gaussian(img2,kernel1=15,kernel2=13,sigma1=0,sigma2=13.0)
        # print('contrast-dog')
        # display(img2)
        # # img2 = cv2.fastNlMeansDenoising(img2, None, 20, 7, 21) 
        # # print('Noise')
        # # display(img2)
        # img2 = gradient_direction(img2)
        # display(img2)
        # img2 = desat_graysc(img2,False)
        # # img2 = up_down_scaling(img2,block_size=8)
        # edge = edge_ascii_image(img2)



        # overlay_images(img,edge)

        # img = cv2.imread(f)
        # img = image_dimension(img)
        # img = enhance_edges(img,saturation=1.12, value=0.6, lightness=1.42)
        # # img = cv2.fastNlMeansDenoising(img, None, 20, 7, 21)
        # # print('Noise')
        # # display(img)
        # img = desat_graysc(img, True)
        # img = prewitt_filter(img)
        # img = difference_of_Gaussian(img,kernel1=15,kernel2=13,sigma1=0,sigma2=15.0)
        # img = gradient_direction(img)
        # img = up_down_scaling(img, block_size=8)
        # # img = desat_graysc(img, True)
        # img = edge_ascii_image(img)
#1st part
        # img = cv2.imread(f)
        # img = image_dimension(img)
        # img = image_sharpen(img)
        # # img = desat_graysc(img,False)
        # # img = enhance_contrast(img)
        # img = difference_of_Gaussian(img,sigma1=0.1,sigma2=6.5)#previouse kernel size (0,0)        
        # img = extended_sobel(img)
        # img = desat_graysc(img,False)
        # # res_up = up_down_scaling(img, block_size= 8)
        # # res_up = desat_graysc(res_up)
        # # print('res_up',res_up.shape)
        # img = process_image_ascii(img)


def gradient_direction(img):
    img = cv2.GaussianBlur(img,(7,7),0)
    Sx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=5)
    Sy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=5)

    Sx = cv2.convertScaleAbs(Sx)
    Sy = cv2.convertScaleAbs(Sy)


    grad_theta = np.atan2(Sy,Sx)/(2*np.pi)

    # print('grad_theta',grad_theta)
    # dic_col = defaultdict(int)  # Initialize with default int (which is 0)
    # for value in grad_theta.flatten():
    #     dic_col[value] += 1
    # print(1,dic_col)
    # display(grad_theta)
    print('Min:',np.min(grad_theta))
    print('Max:',np.max(grad_theta))
    # grad_theta[grad_theta < -0.45]=0

         
    theta_normalise = np.add(grad_theta,0.5,where = grad_theta!=0)    
    theta_image = (theta_normalise*255).astype(np.uint8)
    # theta = defaultdict(int)
    # for pixel_value in theta_image.flatten():
    #     theta[pixel_value] += 1
    # print()
    # print(2,theta)
    theta_image = np.floor(theta_image)
    
    # flo = defaultdict(int)
    # for flo_value in theta_image.flatten():
    #     flo[flo_value] += 1
    # print()
    # print(3,flo)
    # display(theta_image)
    return theta_image

        # edge = cv2.imread(f)
        # edge = image_dimension(edge)
        # edge = enhance_edges(edge,saturation=1.32, value=0.6, lightness=1.22) 
        # edge = cv2.fastNlMeansDenoising(edge, None, 20, 7, 21)        
        # edge = dog(edge,kernel1=15,kernel2=21,sigma1=1.3,sigma2=3.2)
        # # edge = extended_sobel(edge)
        # edge = gradient_direction(edge)
        # edge = desat_graysc(edge,1)              
        # edge = up_down_scaling(edge, block_size= 8)  
        # edge = edge_ascii_image(edge)

        # combi = overlay_images(img,edge)
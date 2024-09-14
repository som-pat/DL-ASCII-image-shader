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
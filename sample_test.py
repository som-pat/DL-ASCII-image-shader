# import cv2
# import numpy as np
# from PIL import Image, ImageFont, ImageDraw

# def apply_pixel_art_effect(image, pixel_size=8):
#     # Step 1: Enlarge the texture by scaling it up by the pixel size factor
#     enlarged_image = cv2.resize(image, (image.shape[1] * pixel_size, image.shape[0] * pixel_size), interpolation=cv2.INTER_NEAREST)
    
#     # Step 2: Floor the values to snap the UVs to the nearest block
#     # Create down-sampled UV coordinates
#     screen_size = np.array([enlarged_image.shape[1], enlarged_image.shape[0]])
#     uv = np.mgrid[0:screen_size[1], 0:screen_size[0]].transpose(1, 2, 0).astype(float) / screen_size
    
#     down_sampled_uv = np.floor(uv * screen_size / pixel_size) / (screen_size / pixel_size)
#     down_sampled_uv = down_sampled_uv.reshape(-1, 2)  # Flatten for sampling

#     # Step 3: Sample the enlarged image using downsampled UV coordinates
#     sampled_texture = enlarged_image[(down_sampled_uv[:, 1] * screen_size[1]).astype(int) % screen_size[1],
#                                      (down_sampled_uv[:, 0] * screen_size[0]).astype(int) % screen_size[0]]
#     sampled_texture = sampled_texture.reshape(screen_size[1], screen_size[0], -1)

#     # Step 4: Downscale back to the original image size
#     final_image = cv2.resize(sampled_texture, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
#     final_image = cv2.rotate(final_image, cv2.ROTATE_90_CLOCKWISE)
#     return final_image

# # Load an image as an example (assuming it's RGB)
# image = cv2.imread('samples/sample.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format

# # Apply the pixel art effect with an 8x8 pixel size
# pixel_size = 8
# result_image = apply_pixel_art_effect(image, pixel_size=pixel_size)

# # Display the resulting image
# cv2.imshow('Pixel Art Effect', result_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

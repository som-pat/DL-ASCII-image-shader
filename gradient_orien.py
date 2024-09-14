import cv2
import numpy as np

def mat2gray(src):
    # Normalize the image to the range 0-255 and convert to 8-bit
    dst = cv2.normalize(src, None, 0, 255, cv2.NORM_MINMAX)
    dst = np.uint8(dst)
    return dst

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

    return ori_map

def main():
    # Create a black image with a white circle
    image = np.zeros((240, 320), dtype=np.uint8)
    cv2.circle(image, (160, 120), 80, (255, 255, 255), -1, cv2.LINE_AA)

    cv2.imshow("original", image)

    # Compute the Sobel gradients
    Sx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    Sy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)

    # Calculate magnitude and orientation of gradients
    mag = cv2.magnitude(Sx, Sy)
    ori = cv2.phase(Sx, Sy, angleInDegrees=True)

    # Create orientation map
    ori_map = orientation_map(mag, ori, thresh=1.0)

    # Show and save results
    cv2.imshow("x", mat2gray(Sx))
    cv2.imshow("y", mat2gray(Sy))
    
    cv2.imwrite("hor.png", mat2gray(Sx))
    cv2.imwrite("ver.png", mat2gray(Sy))

    cv2.imshow("magnitude", mat2gray(mag))
    cv2.imshow("orientation", mat2gray(ori))
    cv2.imshow("orientation map", ori_map)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

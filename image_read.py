import cv2

img = cv2.imread('Dataset/train/Image_2155.jpg')

# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

gray_shade = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.normalize(gray_shade, None, 0, 255, cv2.NORM_MINMAX)

width,height = gray.shape
print(width,height)

def ascii_conversion(gray):
    ascii_ren = []
    ascii_characters = ["@", "#", "S", "%", "?", "*", "+", ";", ":", ","]

    for i in range(gray.shape[0]):
        row = []
        for j in range(gray.shape[1]):
            
            intensity  = gray[i,j]
            intensity_index = int(intensity/25.5)

            try:
                ascii_ch = ascii_characters[intensity_index]
            except:
                print(intensity_index)
                break
            row.append(ascii_ch)
        ascii_ren.append(row)

    for row in ascii_ren:
        print("".join(row))

# ascii_conversion(gray)

print(img.shape)

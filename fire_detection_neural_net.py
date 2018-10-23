'''
Fire detection
Algorithm :
1) Pre processing images (reshape image to a fixed size )
2) Create labels (one-hot)
3) Check with various models
4) Check time requirement
'''
import numpy as np
import cv2
import os

#loading images for folder
pos_folder = 'Dataset_temp/pos_image_temp/'
neg_folder = 'Dataset_temp/neg_image_temp/'

#load images into array
'''
Things to do
1) calculate dimension of smallest image
2) resize image to same dimesnion
3)
'''
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if any([filename.endswith(x) for x in ['.jpg']]):
            img = cv2.imread(os.path.join(folder , filename))

            if img is not None:
                images.append(img)
    return images
def resize_image(images):
    x = 10000
    y = 10000
    #find the image with minimum dimesnion
    for img in images:
        temp_x = np.shape(img)[0]
        temp_y = np.shape(img)[1]
        if x > temp_x:
            x = temp_x
        if y > temp_y:
            y = temp_y
    img_resized = []
    #for having constant image dimension
    for img in images:
        img = img[:x+1 , :y+1]
        img_resized.append(img)
    return img_resized
pos_images  = load_images_from_folder(pos_folder)
pos_images_resized = resize_image(pos_images)
neg_images = load_images_from_folder(neg_folder)
neg_images_resized = resize_image(neg_images)
print("Positive images dimension:", np.shape(pos_images))
print("Begative images dimesnion:",np.shape(neg_images))

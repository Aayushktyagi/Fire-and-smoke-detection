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
1) calculate dimension of smallest image(Done)
2) resize image to same dimesnion(Done)
3) Vectorize image and create an array(Done)
4) Concatenate both pos and negative array_to_image
5) Create labels(one-hot encoding )
'''
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if any([filename.endswith(x) for x in ['.jpg']]):
            img = cv2.imread(os.path.join(folder , filename))

            if img is not None:
                images.append(img)
    return images
x_min = 320
y_min = 240
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
        img = img[:y_min , :x_min,:]
        img_resized.append(img)
    return img_resized

def vector_array(images):
    #chnage input array dimension based on image shape
    img_vector = np.zeros(shape=(230400,1))
    for idx , img in enumerate(images):
        img_vec = np.reshape(img,-1)
        img_vec = img_vec[:,np.newaxis]
        img_vector = np.append(img_vector,img_vec, axis = 1)
        #print("Shape of image vector {} ,{}".format(idx,np.shape(img_vec)))
    print("Shape of img_vector :",np.shape(img_vector))
    return img_vector


pos_images  = load_images_from_folder(pos_folder)
pos_images_resized = resize_image(pos_images)
neg_images = load_images_from_folder(neg_folder)
neg_images_resized = resize_image(neg_images)
pos_image_array = vector_array(pos_images_resized)
neg_image_array = vector_array(neg_images_resized)
print("Positive images dimension:", np.shape(pos_image_array))
print("Negative images dimesnion:",np.shape(neg_image_array))

pos_label = np.ones(shape=(np.shape(pos_image_array)[1],1))
print("pos labels shape{}".format(np.shape(pos_label)))
neg_labels = np.zeros(shape=(np.shape(neg_image_array)[1],1))
print("neg labels shape{}".format(np.shape(neg_labels)))

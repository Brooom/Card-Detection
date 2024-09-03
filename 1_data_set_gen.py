import time
import glob
import random
from random import randint
import numpy as np
import pandas as pd
import cv2
import os
from library.img_rando import img_zoom_rand, img_rot_rand, img_3D_rand, img_pos_rand, img_blure
from library.config import PATH_CARDS, PATH_BACKGROUND, PATH_ALL_IMAGES, CSV_NAME, DATASET_NAME, SPLIT_PATH

# Set number of differnt iterations per image
NUM_ITERATIONS = 150
random.seed('train')

# Get all the card images
print("Card Images:")
images = np.empty((len(glob.glob(f"{PATH_CARDS}/*.png")),400,257,3), dtype="uint8")
for i, file in enumerate(glob.glob(f"{PATH_CARDS}/*.png")):
    img = cv2.imread(file)
    img[img==0] = 1
    images[i,:,:,:] = np.asarray(img)

    # Print filename withou path and extension
    filename = file.split("\\")[-1].split(".")[0]
    print(f"{i}: {filename}")

images = np.array(images)
images[images[:,:,:,:] == 0] = 1 # Image is  not allowed to have any 0 vales at the beginning

back_images = np.empty((len(glob.glob(PATH_BACKGROUND+"/*.jpg")),1000,1500,3), dtype="uint8")
for i, file in enumerate(glob.glob(PATH_BACKGROUND+"/*.jpg")):
    back_img = cv2.imread(file)
    if back_img.shape != (1000,1500,3):
        back_img = cv2.resize(back_img, (1500, 1000))
    back_images[i,:,:,:] = back_img

#Output some info
print(f"\nNumber of background images: {back_images.shape[0]}")
print(f"Number of imges to be generated per card: {NUM_ITERATIONS}")
print(f"Number of images to be generated in total: {NUM_ITERATIONS*images.shape[0]}")
labels = np.array([('filename','width','height','class','xmin','ymin','xmax','ymax')])
count = 0

# Check if the all images folder is empty   
if len(os.listdir(PATH_ALL_IMAGES)) != 0:
    print("All images folder is not empty. Please empty the folder before generating new images.")
    exit()

# Generate Data
start_time = time.time()
for img, index in zip(images, range(0, images.shape[0])):

    print("\n")

    for i in range(NUM_ITERATIONS):

        img_zoom = img_zoom_rand(img, maxZoom=2.1)
        img_rot = img_rot_rand(img_zoom)
        img_pov = img_3D_rand(img_rot)

        rand = randint(0, back_images.shape[0]-1)
        back = back_images[rand,:,:,:] # Random background image
        img_pos, pos = img_pos_rand(img_pov,back)

        kernel_sizes = [1,3,5] # Get random kernel size/blur
        kernel_size = kernel_sizes[randint(0,2)]
        final_img = img_blure(img_pos, kernel_size)

        img_pov_color = img_pov.copy()
        img_pov_gray = cv2.cvtColor(img_pov, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(img_pov_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        rect = cv2.boundingRect(contours[0])

        # Save image
        image_filename = 'img_' + str(index).rjust(2,"0") + "_"+str(count).rjust(6,"0") + ".png"
        image_path = PATH_ALL_IMAGES + image_filename
        print(image_filename)
        cv2.imwrite(image_path, (final_img*255).astype(np.uint8))
        labels = np.append(labels,[(image_filename, final_img.shape[1], final_img.shape[0], index,pos[0] + rect[0], pos[1] + rect[1], rect[0] + rect[2] + pos[0], rect[1] + rect[3] + pos[1])], 0)
        count += 1

end_time = time.time()
print(f"Total time taken: {round(end_time-start_time)}s")

# Save labels
np.savetxt(CSV_NAME, labels, delimiter=",", fmt='%s')


############################################################################################################

# Generate labels
data = pd.read_csv(CSV_NAME)
print(f"Number of images: {len(data)}")

for i in range(len(data)):
    label=[] #class center_x center_y width height - every mesurement is relative to the image size
    print(f"Gernerating label for image {i}.")
    img_class = data.iloc[i]['class']
    center_x = (data.iloc[i]['xmin']+data.iloc[i]['xmax'])/2
    center_y = (data.iloc[i]['ymin']+data.iloc[i]['ymax'])/2
    img_width = data.iloc[i]['width']
    img_height = data.iloc[i]['height']
    label = np.append(label,[img_class, round(center_x/img_width,5), round(center_y/img_height,5), round((data.iloc[i]['xmax']-data.iloc[i]['xmin'])/img_width,5), round((data.iloc[i]['ymax']-data.iloc[i]['ymin'])/img_height,5)])
    np.savetxt(PATH_ALL_IMAGES + data.iloc[i]['filename'][0:-4]+".txt", label, newline=" ",fmt='%s')

# Generate dataset.yaml
with open(DATASET_NAME, 'w', encoding='utf-8') as file:
    file.write(f"path: ./{SPLIT_PATH}\n")
    file.write("train: ./train\n")
    file.write("val: ./validation\n")
    file.write("test: ./test\n")
    file.write("\n")
    file.write("# Class names\n")
    file.write("names:\n")

    for i, filename in enumerate(glob.glob(f"{PATH_CARDS}/*.png")):
        class_name = filename.split("\\")[-1].split(".")[0]
        file.write(f"  {i}: {class_name}\n")

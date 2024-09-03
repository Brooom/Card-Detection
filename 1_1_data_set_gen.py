import time
from random import randint
import numpy as np
import cv2
from library.img_rando import *

# Train
random.seed('train')
PATH_ALL_IMAGES = '1_all_images/'
CSV_NAME = '1_all_images/all_labels'

# Set number of differnt iterations (min 1)
NUM_ITERATIONS = 100

# import images and cvs
PATH_CARDS = '0_cards_images'
PATH_BACKGROUND = '0_background_images'

# Create a list of all the card images
print("Card Images:")

images = np.empty((len(glob.glob(PATH_CARDS+"/*.png")),400,257,3), dtype="uint8")
i=0
for file in glob.glob(PATH_CARDS+"/*.png"):
    img = cv2.imread(file)
    img[img==0] = 1
    images[i,:,:,:] = np.asarray(img)
    # Print filename withou path and extension
    filename = file.split("\\")[-1].split(".")[0]
    print(f"{i}: {filename}")
    i+=1

images = np.array(images)
images[images[:,:,:,:]==0] = 1 # Image is  not allowed to have any 0 vales at the beginning

print("#########################################")
print("Background Images:")
back_images = np.empty((len(glob.glob(PATH_BACKGROUND+"/*.jpg")),1000,1500,3), dtype="uint8")
i=0
for file in glob.glob(PATH_BACKGROUND+"/*.jpg"):
    back_img = cv2.imread(file)
    if back_img.shape != (1000,1500,3):
        back_img = cv2.resize(back_img, (1500, 1000))
    back_images[i,:,:,:] = back_img
    i+=1

#Output some info
print(f"Backgrounds: {back_images.shape}")
print(f"Number of imges to be generated per card: {NUM_ITERATIONS}")
print(f"Number of images to be generated in total: {NUM_ITERATIONS*images.shape[0]}")
labels = np.array([('filename','width','height','class','xmin','ymin','xmax','ymax')])
count = 0

# Generate Data
start = time.time()
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
        image_filename = 'img_'+str(index).rjust(2,"0")+"_"+str(count).rjust(6,"0")+".png"
        image_path = PATH_ALL_IMAGES + image_filename
        print(image_filename)
        cv2.imwrite(image_path, (final_img*255).astype(np.uint8))
        labels = np.append(labels,[(image_filename, final_img.shape[1],final_img.shape[0],index,pos[0]+rect[0],pos[1]+rect[1],rect[0]+rect[2]+pos[0],rect[1]+rect[3]+pos[1])],0)
        count += 1

end = time.time()
print(f"Total time taken: {round(end-start)}s")

# Save labels
np.savetxt(f"{CSV_NAME}.csv", labels, delimiter=",", fmt='%s')

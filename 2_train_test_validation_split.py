import os
import fnmatch
import shutil
import random
from library.config import PATH_ALL_IMAGES, PATH_TRAIN, PATH_TEST, PATH_VALIDATION

image_datatypes = ['jpg', 'png', 'jpeg', 'bmp']
nr_train = 0
nr_test = 0
nr_validation = 0

# Check if the train folder exists
if not os.path.exists(PATH_TRAIN):
    os.makedirs(PATH_TRAIN)

for root, dir, files in os.walk(PATH_TRAIN):
    if files != []:
        raise Exception("Train folder is not empty.")

# Check if the test folder exists
if not os.path.exists(PATH_TEST):
    os.makedirs(PATH_TEST)

for root, dir, files in os.walk(PATH_TEST):
    if files != []:
        raise Exception("Test folder is not empty.")

# Check if the validation folder exists
if not os.path.exists(PATH_VALIDATION):
    os.makedirs(PATH_VALIDATION)

for root, dir, files in os.walk(PATH_VALIDATION):
    if files != []:
        raise Exception("Validation folder is not empty.")

for root, dir, files in os.walk(PATH_ALL_IMAGES):
    for items in fnmatch.filter(files, "*"):
        if items[-3:len(items)] in image_datatypes:
            random_number = random.random()
            print(PATH_TRAIN+items)
            print(PATH_TRAIN+items[0:-4]+".txt")
            if 0 < random_number < 0.6:
                shutil.move(PATH_ALL_IMAGES+items, PATH_TRAIN+items)
                shutil.move(PATH_ALL_IMAGES+items[0:-4]+".txt", PATH_TRAIN+items[0:-4]+".txt")
                nr_train += 1

            elif random_number < 0.8:
                shutil.move(PATH_ALL_IMAGES+items, PATH_VALIDATION+items)
                shutil.move(PATH_ALL_IMAGES+items[0:-4]+".txt", PATH_VALIDATION+items[0:-4]+".txt")
                nr_validation += 1

            else:
                shutil.move(PATH_ALL_IMAGES+items, PATH_TEST+items)
                shutil.move(PATH_ALL_IMAGES+items[0:-4]+".txt", PATH_TEST+items[0:-4]+".txt")
                nr_test += 1

print(f"Number of train images: {nr_train}")
print(f"Number of test images: {nr_test}")
print(f"Number of validation images: {nr_validation}")

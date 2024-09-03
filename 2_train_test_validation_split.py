import os
import fnmatch
import shutil
import random

all_images = '1_all_images/'
path_train = '2_splited_data/train/'
path_test = '2_splited_data/test/'
path_validation = '2_splited_data/validation/'
image_datatypes = ['jpg', 'png', 'jpeg', 'bmp']
nr_train= 0
nr_test= 0
nr_validation= 0

# Check if the train folder exists
if not os.path.exists(path_train):
    os.makedirs(path_train)

for root, dir, files in os.walk(path_train):
    if files != []:
        raise Exception("Train folder is not empty.")

# Check if the test folder exists
if not os.path.exists(path_test):
    os.makedirs(path_test)

for root, dir, files in os.walk(path_test):
    if files != []:
        raise Exception("Test folder is not empty.")

# Check if the validation folder exists
if not os.path.exists(path_validation):
    os.makedirs(path_validation)

for root, dir, files in os.walk(path_validation):
    if files != []:
        raise Exception("Validation folder is not empty.")

for root, dir, files in os.walk(all_images):
    for items in fnmatch.filter(files, "*"):
        if items[-3:len(items)] in image_datatypes:
            random_number = random.random()
            print(path_train+items)
            print(path_train+items[0:-4]+".txt")
            if 0 < random_number < 0.6:
                shutil.move(all_images+items, path_train+items)
                shutil.move(all_images+items[0:-4]+".txt", path_train+items[0:-4]+".txt")
                nr_train += 1

            elif random_number < 0.8:
                shutil.move(all_images+items, path_validation+items)
                shutil.move(all_images+items[0:-4]+".txt", path_validation+items[0:-4]+".txt")
                nr_validation += 1

            else:
                shutil.move(all_images+items, path_test+items)
                shutil.move(all_images+items[0:-4]+".txt", path_test+items[0:-4]+".txt")
                nr_test += 1

print(f"Number of train images: {nr_train}")
print(f"Number of test images: {nr_test}")
print(f"Number of validation images: {nr_validation}")

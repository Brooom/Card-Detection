#%%
import os
import fnmatch
import shutil
import random 
#%%

all_images = '1_all_images/'
path_train = '2_splited_data/train/'
path_test = '2_splited_data/test/'
path_validation = '2_splited_data/validation/'
image_datatypes = ['jpg', 'png', 'jpeg', 'bmp']
nr_train= 0
nr_test= 0
nr_validation= 0

for root, dir, files in os.walk(path_train):
    if files != []:
        raise Exception("Train folder is not empty.")

for root, dir, files in os.walk(path_test):
    if files != []:
        raise Exception("Test folder is not empty.")

for root, dir, files in os.walk(path_validation):
    if files != []:
        raise Exception("Validation folder is not empty.")


for root, dir, files in os.walk(all_images):
    for items in fnmatch.filter(files, "*"):
        if items[-3:len(items)] in image_datatypes:
            random_number = random.random()
            print(path_train+items)
            print(path_train+items[0:-4]+".txt")
            if random_number > 0 and random_number < 0.6:
                shutil.move(all_images+items, path_train+items)
                shutil.move(all_images+items[0:-4]+".txt", path_train+items[0:-4]+".txt")
                nr_train+=1
            elif random_number > 0.6 and random_number < 0.8:
                shutil.move(all_images+items, path_test+items)
                shutil.move(all_images+items[0:-4]+".txt", path_test+items[0:-4]+".txt")
                nr_test+=1
            elif random_number > 0.8 and random_number < 1:
                shutil.move(all_images+items, path_validation+items)
                shutil.move(all_images+items[0:-4]+".txt", path_validation+items[0:-4]+".txt")
                nr_validation+=1
            
print("Nr of train images: ", nr_train)
print("Nr of test images: ", nr_test)
print("Nr of validation images: ", nr_validation)
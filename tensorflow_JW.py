# Tutorial by https://medium.com/swlh/image-classification-for-playing-cards-26d660f3149e
# https://www.tensorflow.org/tutorials/train_images/cnn
#%%
from turtle import back
import tensorflow as tf
print (tf.config.list_physical_devices()) #run to make sure tensorflow is connected to gpu
#print (tf.test.is_gpu_available())
from tensorflow.keras.utils import Sequence

import numpy as np
import pandas as pd
import cv2  
import os  
from random import shuffle
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread, imshow, subplots, show
import sys
import glob
from datetime import datetime



sys.path.append('D:/FHGR/05_HS_22/BiV_02/00_Projekt/Biv2')
sys.path.append('W:\FHGR\Biv2')


#%% 
# import images and cvs

train_path = 'train_set'
test_path = 'test_set'
path = 'Cards'
card_list_dtype = (int,str,str,ord,int)

#%% 
# load cards/class
card_list = np.loadtxt('card_list.csv',delimiter=";", dtype=str)[1:]
print(len(card_list))
#remove ignord card 
card_list = np.delete(card_list,card_list[:,-1]!='0',axis=0)
print(len(card_list))

if len(glob.glob(path+"/*.png"))!=len(card_list):
    raise Exception('number of Cards ({}) not equel to enabled cards ({})'.format(len(glob.glob(path+"/*.png")),len(card_list)))
        
class_names = list(card_list[:, 1])

#%% 
# load test set
test_labels = np.loadtxt('test_labels.csv',delimiter=",", dtype=str)[1:]
test_labels = np.array(test_labels[:,3],dtype=int)-1

test_images = [cv2.imread(file,0) for file in glob.glob(test_path+"/*.png")]
test_images = np.array(test_images)

if test_images.shape[0] != len(test_labels):
    raise Exception('number of test imges ({}) not equel to enabled test labels ({})'.format(test_images.shape[0],len(test_labels)))

print("loadid test data succefuly")
#%% 
# load training labels
train_labels = np.loadtxt('train_labels.csv',delimiter=",", dtype=str)[1:]
train_labels = np.array(train_labels[:,3],dtype=int)-1

print("loaded training labels",len(train_labels),"succefuly")
print("going to load         ",len(glob.glob(train_path+"/*.png")),"test images")
#%%
# load training images
train_images = [cv2.imread(file,0) for file in glob.glob(train_path+"/*.png")]
train_images = np.array(train_images)

if train_images.shape[0] != len(train_labels):
    raise Exception('number of train imges ({}) not equel to enabled train labels ({})'.format(train_images.shape[0],len(train_labels)))

print("loaded training data succefuly")
#%% 
# shuffel
print("shuffel traing data")
np.random.seed(1)
np.random.shuffle(train_labels)
np.random.seed(1)
np.random.shuffle(train_images)

print("shuffel test data")
np.random.seed(2)
np.random.shuffle(test_labels)
np.random.seed(2)
np.random.shuffle(test_images)
print("shuffel complet")

#%% debug
# display example of shuffeld (the first 25)
# 
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.gray()
#     plt.imshow(train_images[i])
#     plt.xlabel(class_names[train_labels[i]])
    
# plt.show()

#%%
# prep data

batch_size=8

class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

# force tensor flow to load data in batches
train_gen = DataGenerator(train_images, train_labels, batch_size)
test_gen = DataGenerator(test_images, test_labels, batch_size)

#%%
# init module
tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

epochs=7


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(514, 800,1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

model.summary()

#%% load checkpoint file if existing
#todo  look at all cp file an load latest / fitting one
#%%
# training module
cp_name = "02_"+str(epochs).rjust(2,"0")+"_10epochs_conv.h5"
print(cp_name)
cp = tf.keras.callbacks.ModelCheckpoint(filepath=cp_name,
                               save_best_only=True,
                               verbose=1)

model.compile(loss = 'sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit(train_gen, epochs=epochs, batch_size=batch_size,validation_data=test_gen, callbacks=[cp]).history

#%%
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# history = model.fit(train_images, train_labels, epochs=10, 
#                     validation_data=(test_images, test_labels))
#%% display traning data

acc = history['accuracy']
val_acc = history['val_accuracy']
loss = history['loss']
val_loss = history['val_loss']
#epochs = range(len(acc)) epochs is all redy correct

plt.figure(figsize = (12,8))
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)


plt.figure(figsize = (12,8))
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)

plt.show()

# %% save module

now = datetime.now()
module_path ="my_model_" + now.strftime("%d%m_%H%M")

model.save(module_path)
# us the .h5 file or this to reload

# df=pd.read_csv('card_labels.csv')
# labels=list(df['label'])

# predictions=model.predict_classes(test_X)

# sample=test_X[:16]

# plt.figure(figsize=(16,16))
# for i in range(16):
#     plt.subplot(4,4,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(sample[i].reshape(sample.shape[1], sample.shape[2]))
#     plt.xlabel(labels[predictions[i]])
# plt.show()
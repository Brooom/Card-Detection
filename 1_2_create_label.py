import numpy as np
import os
import pandas as pd
import cv2

data=pd.read_csv('1_all_images/all_labels.csv')
path='1_all_images/'

print("Nr of images: ",len(data))
for i in range(len(data)):
    label=[] #class center_x center_y width height     - every mesurement is relative to the image size
    print(i)
    center_x = (data.iloc[i]['xmin']+data.iloc[i]['xmax'])/2
    center_y = (data.iloc[i]['ymin']+data.iloc[i]['ymax'])/2
    img_width = data.iloc[i]['width']
    img_height = data.iloc[i]['height']
    label=np.append(label,[1, round(center_x/img_width,5), round(center_y/img_height,5), round((data.iloc[i]['xmax']-data.iloc[i]['xmin'])/img_width,5), round((data.iloc[i]['ymax']-data.iloc[i]['ymin'])/img_height,5)])
    np.savetxt(path+data.iloc[i]['filename'][0:-4]+".txt", label, newline=" ",fmt='%s')
    #img = cv2.imread(path+data.iloc[i]['filename'])
    #cv2.rectangle(img,(int(data.iloc[i]['xmin']),int(data.iloc[i]['ymin'])),(int(data.iloc[i]['xmax']),int(data.iloc[i]['ymax'])),(0,255,0),2)
    #print(label)
    #cv2.imshow('img',img)
    #cv2.waitKey(0)

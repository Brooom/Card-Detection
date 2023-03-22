#%%
import tensorflow as tf

import numpy as np
import cv2 
import glob
import os

#show devices connected to tensorflow 
#debug print (tf.config.list_physical_devices()) 

# function to clean cmd in windows 
clearConsole = lambda: os.system('cls' if os.name in ('nt', 'dos') else 'clear')


# varibles ues throut the programm
path_model = '02_07_epochs_conv.h5'
path_card = 'Cards'
dim = (800, 514) # img size of ai imput
max = [0,0,0,0] #max values of prediction
max_i = [0,0,0,0] # index of max values of prediction


#%% 
# load cards/class
card_list = np.loadtxt('card_list.csv',delimiter=";", dtype=str)[1:] #load without header
#debug print(len(card_list))
#remove ignord card 
card_list = np.delete(card_list,card_list[:,-1]!='0',axis=0)
#debug print(len(card_list))

if len(glob.glob(path_card+"/*.png"))!=len(card_list):
    raise Exception('number of Cards ({}) not equel to enabled cards ({})'.format(len(glob.glob(path_card+"/*.png")),len(card_list)))
        
class_names = list(card_list[:, 1])

#%%
# load ai model
new_model = tf.keras.models.load_model('5epochs_conv.h5')
clearConsole()

def predict(_model,_img,_verbos=1):
    """make a prediction with the given tf model and image

    Args:
    ------
        _model (tf.moddel): a trained tensorflow model
        _img (np.array(dtype=uinit8)): a Grayscale imge of size 514,800
        _verbos (int, optional): Verbose level of of the predict funktion. Defaults to 1.
    """
    pred =_model.predict(_img.reshape(-1,514,800),verbose= _verbos)

    #if prediction is not 100% display the the top 3
    if(pred.max() != 1.0):
        
        for i in range(0,3):         
            print(class_names[pred.argmax()],pred.max())
            pred[0,pred.argmax()] = 0
    else:
        print(class_names[pred.argmax()],pred.max())


cv2.namedWindow("static")
cv2.waitKey(0)

#%% 
# static demo

for name in glob.glob('sample/img_*.png'):
    
    img = cv2.imread(name,0)

    cv2.imshow('static', img)
    
    #debug print(name)
    predict(new_model,img)

    print()
    cv2.waitKey(0) # wait for the next image

cv2.destroyWindow('static')



#%%
# open webcam
cv2.namedWindow("video")
cam = cv2.VideoCapture(2)

while True:

    
    #display a preprocecd webcam frame
    check, frame = cam.read()

    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('video', frame)

    #read key an do action
    key = cv2.waitKey(1)
    if key == 27: # esc
        break

    elif key == 113: # q
        predict(new_model,img)
        cv2.waitKey(250) 
        print()

#clean up
cam.release()
cv2.destroyAllWindows()

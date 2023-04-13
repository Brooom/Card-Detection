#%%
from library.img_rando import *
import time
from random import randint
import numpy as np
import cv2

#%%
# train 
random.seed('train')

path_all_images = '1_all_images/'

csv_name = '1_all_images/all_labels'

#set number of differnt inerations (min 1)

zoom_data_size= 5
rot_data_size = 3
POV_data_size = 3
pos_data_size = 3

# import images and cvs
path_cards = '0_cards_images'
path_background = '0_background_images'

#creates a list of all the card images

images = np.empty((len(glob.glob(path_cards+"/*.png")),400,257,3), dtype="uint8") 
i=0
for file in glob.glob(path_cards+"/*.png"):
    img=cv2.imread(file)
    img[img==0]=1
    images[i,:,:,:]=np.asarray(img)
    print("#########################################")
    print("Labels")
    print("Index: "+ str(i))
    print("Name: "+file)
    print("#########################################")
    i+=1


images = np.array(images)
images[images[:,:,:,:]==0]=1 #Image is  not allowed to have any 0 vales at the beginning

back_images = np.empty((len(glob.glob(path_background+"/*.jpg")),1000,1500,3), dtype="uint8") 
i=0
for file in glob.glob(path_background+"/*.jpg"):
    back_img = cv2.imread(file)
    if back_img.shape != (1000,1500,3):
        back_img = cv2.resize(back_img, (1500, 1000))
    back_images[i,:,:,:]=back_img
    i+=1


#%%

#Output some info
print("Backgrounds: ",back_images.shape)
print("Num img going to bo gererated per card: ",zoom_data_size* rot_data_size* POV_data_size*pos_data_size*3)
print("Num img going to bo gererated in total: ",zoom_data_size* rot_data_size* POV_data_size*pos_data_size*3*images.shape[0])



labels = np.array([('filename','width','height','class','xmin','ymin','xmax','ymax')])
count = 0

#%%
# Generate Data

start = time.time()
#print(path_all_images)
for img, index in zip(images,range(0,images.shape[0])):
    
    print("img_" ,index)

    for i_zoom in range(zoom_data_size):
        img_zoom = img_zoom_rand(img, maxZoom=2.1)
        #cv2.imshow("img_zoom", img_zoom)
        #cv2.waitKey(0)

        for i_rot in range(rot_data_size):
            img_rot = img_rot_rand(img_zoom)
            #cv2.imshow("img_rot", img_rot)
            #cv2.waitKey(0)

            for i_pov in range(POV_data_size):
                img_pov = img_3D_rand(img_rot)
                #cv2.imshow("img_pov", img_pov)
                #cv2.waitKey(0)
                for i_pos in range(pos_data_size):
                    rand=randint(0,back_images.shape[0]-1)
                    back = back_images[rand,:,:,:]     
                    img_pos,pos = img_pos_rand(img_pov,back)
                    #cv2.imshow("img_pos", img_pos)
                    #cv2.waitKey(0)

                    for kernel_size in [1,3,5]:
                        final_img=img_blure(img_pos,kernel_size)

                        img_pov_color=img_pov.copy()
                        img_pov_gray=cv2.cvtColor(img_pov, cv2.COLOR_BGR2GRAY)
                        contours, hierarchy=cv2.findContours(img_pov_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE            )

                        rect=cv2.boundingRect(contours[0])

                        #cv2.imshow("final_img", (final_img*255).astype(np.uint8))
                        #cv2.waitKey(0)

                        #save image
                        print(path_all_images+'img_'+str(index).rjust(2,"0")+"_"+str(count).rjust(6,"0")+".png")
                        cv2.imwrite(path_all_images+'img_'+str(index).rjust(2,"0")+"_"+str(count).rjust(6,"0")+".png",(final_img*255).astype(np.uint8))
                        labels=np.append(labels,[('img_'+str(index).rjust(2,"0")+"_"+str(count).rjust(6,"0")+".png",
                                                            final_img.shape[1],final_img.shape[0],index,pos[0]+rect[0],pos[1]+rect[1],rect[0]+rect[2]+pos[0],rect[1]+rect[3]+pos[1])],0)
                        count = count +1
end = time.time()
print("Time: "+ str(end - start))


np.savetxt(csv_name+".csv", labels, delimiter=",",fmt='%s')

cv2.destroyAllWindows()




# %%

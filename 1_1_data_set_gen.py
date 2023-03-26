#%%
from library.img_rando import *
import time

#%%
# train 
random.seed('train')

path_all_images = '1_all_images/'

csv_name = '1_all_images/all_labels'

#set number of differnt inerations (min 1)
master_data_size = 2
zoom_data_size= master_data_size
rot_data_size = master_data_size
POV_data_size = master_data_size
pos_data_size = master_data_size

# import images and cvs
path_cards = '0_cards_images'
path_background = '0_background_images'

#creates a list of all the card images
images = [cv2.imread(file,0) for file in glob.glob(path_cards+"/*.png")] 
images = np.array(images)
images[images==0]=1 #Image is  not allowed to have any 0 vales at the beginning

back_images = [cv2.imread(file,0) for file in glob.glob(path_background+"/*.jpg")] #[7:]
back_images = np.array(back_images)

#Output some info
print("Backgrounds: ",back_images.shape)
print("Num img going to bo gererated per card: ",back_images.shape[0]*zoom_data_size* rot_data_size* POV_data_size*pos_data_size )
print("Num img going to bo gererated in total: ",back_images.shape[0]*zoom_data_size* rot_data_size* POV_data_size*pos_data_size*images.shape[0])



labels = np.array([('filename','width','height','class','xmin','ymin','xmax','ymax')])
count = 0

#%%
# Generate Data

start = time.time()
#print(path_all_images)
for img, index in zip(images,range(1,images.shape[0]+1)):
    
    #debug if index == 2: 
        #debug break 
    print("img_" ,index)

    for i_zoom in range(zoom_data_size):
        img_zoom = img_zoom_rand(img)
        print(1)
        for i_rot in range(rot_data_size):
            img_rot = img_rot_rand(img_zoom)
            print(2)
            for i_pov in range(POV_data_size):
                img_pov = img_3D_rand(img_rot)

                print(3)
                for back in back_images:
                    for i_pos in range(pos_data_size):
                        img_pos,pos = img_pos_rand(img_pov,back)
                        for kernel_size in [1,3,5]:
                            final_img=img_blure(img_pos,kernel_size)


                            contours, hierarchy=cv2.findContours(img_pov, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE            )
                            img_pov_color=cv2.cvtColor(img_pov, cv2.COLOR_GRAY2BGR)
                            #cv2.drawContours(img_pov_color, contours, -1, (0,255,0), 3)
                            #cv2.imshow('img',img_pov_color)
                            #cv2.waitKey(0)
                            rect=cv2.boundingRect(contours[0])
                            #cv2.rectangle(img_pov_color,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(0,255,0),2)
                            #cv2.imshow('img',img_pov_color)
                            #cv2.waitKey(0)

                            #save image
                            print(path_all_images+'img_'+str(index).rjust(2,"0")+"_"+str(count).rjust(6,"0")+".png")
                            cv2.imwrite(path_all_images+'img_'+str(index).rjust(2,"0")+"_"+str(count).rjust(6,"0")+".png",final_img)
                            labels=np.append(labels,[('img_'+str(index).rjust(2,"0")+"_"+str(count).rjust(6,"0")+".png",
                                                            final_img.shape[1],final_img.shape[0],index,pos[0]+rect[0],pos[1]+rect[1],rect[0]+rect[2]+pos[0],rect[1]+rect[3]+pos[1])],0)


                            final_img_color=cv2.cvtColor(final_img, cv2.COLOR_GRAY2BGR)
                            #cv2.rectangle(final_img_color,(pos[0]+rect[0],pos[1]+rect[1]),(rect[0]+rect[2]+pos[0],rect[1]+rect[3]+pos[1]),(0,255,0),2)
                            #cv2.imshow('img',final_img_color)
                            #cv2.waitKey(0)
                            count = count +1
end = time.time()
print("Time: "+ str(end - start))


np.savetxt(csv_name+".csv", labels, delimiter=",",fmt='%s')

#cv2.waitKey(1000)
cv2.destroyAllWindows()
#np.append(test,[images[1]],0)



# %%

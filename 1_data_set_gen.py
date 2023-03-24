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
path_cards = '1_cards_images'
path_background = '1_background_images'

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



labels = np.array([('filename','with','height','class','xmin','ymin','xmax','ymax')])
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
                        
                        #save image
                        print(path_all_images+'img_'+str(index).rjust(2,"0")+"_"+str(count).rjust(6,"0")+".png")
                        cv2.imwrite(path_all_images+'img_'+str(index).rjust(2,"0")+"_"+str(count).rjust(6,"0")+".png",img_pos)
                        labels=np.append(labels,[('img_'+str(index).rjust(2,"0")+"_"+str(count).rjust(6,"0")+".png",
                                                        img_pos.shape[0],img_pos.shape[1],index,pos[0],pos[1],pos[0]+img_pos.shape[1],pos[1]+img_pos.shape[0])],0)

                        count = count +1
end = time.time()
print("Time: "+ str(end - start))


np.savetxt(csv_name+".csv", labels, delimiter=",",fmt='%s')

#cv2.waitKey(1000)
cv2.destroyAllWindows()
#np.append(test,[images[1]],0)



# %%

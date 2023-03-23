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

#%%
# load csv, images and background + create masks
#sys.path.append('D:/FHGR/05_HS_22/BiV_02/00_Projekt/Biv2')
#sys.path.append('W:\FHGR\Biv2')

# import images and cvs

path_cards = '1_cards_images'
path_background = '1_background_images'

#creates a list of all the card images
images = [cv2.imread(file,0) for file in glob.glob(path_cards+"/*.png")] #[:4]  
images = np.array(images)

back_images = [cv2.imread(file,0) for file in glob.glob(path_background+"/*.jpg")] #[7:]
back_images = np.array(back_images)

#Output some info
print("Backgrounds: ",back_images.shape)
print("Num img going to bo gererated per card: ",back_images.shape[0]*zoom_data_size* rot_data_size* POV_data_size*pos_data_size )
print("Num img going to bo gererated in total: ",back_images.shape[0]*zoom_data_size* rot_data_size* POV_data_size*pos_data_size*images.shape[0])


#card_list = np.loadtxt('card_list.csv',delimiter=";", dtype=str)[1:]
#debug for col in range(card_list.shape[1]):
#debug     print(card_list_dtype[col])

masks = images.copy()
#debug img_kill( cv2.rectangle(masks[1],[10,10],[247,390],255,-1))
#debug img_kill(images[1])
for mask in masks:
    mask = cv2.rectangle(mask,[10,10],[247,390],255,-1)
    cv2.threshold(mask,80,255,cv2.THRESH_BINARY,mask)

#debug for mask in masks:  
#debug     cv2.imshow("test",mask)
#debug     cv2.waitKey(0)


labels = np.array([('filename','with','height','class','xmin','ymin','xmax','ymax')])
count = 0

#%%
# Generate Data

start = time.time()
#print(path_all_images)
for img, mask, index in zip(images,masks,range(1,images.shape[0]+1)):
    
    #debug if index == 2: 
        #debug break 
    print("img_" ,index)

    for i_zoom in range(zoom_data_size):
        print(type(img.sum()+i_zoom))
        img_zoom = img_zoom_rand(img,img.sum()+i_zoom)
        mask_zoom = img_zoom_rand(mask,img.sum()+i_zoom)
        print(1)
        for i_rot in range(rot_data_size):
            img_rot = img_rot_rand(img_zoom,img_zoom.sum()+i_rot)
            mask_rot = img_rot_rand(mask_zoom,img_zoom.sum()+i_rot)
            print(2)
            for i_pov in range(POV_data_size):
                img_pov = img_3D_rand(img_rot,img_rot.sum()+i_pov)
                mask_pov = img_3D_rand(mask_rot,img_rot.sum()+i_pov)
                print(3)
                for back in back_images:
                    for i_pos in range(pos_data_size):
                        img_pos,pos = img_pos_rand(img_pov,back,mask_pov, img_pov.sum()+i_pos+back.sum())
                        #debug img_kill(img_pos,5)
                        
                        #save image
                        print(path_all_images+'img_'+str(index).rjust(2,"0")+"_"+str(count).rjust(6,"0")+".png")
                        cv2.imwrite(path_all_images+'img_'+str(index).rjust(2,"0")+"_"+str(count).rjust(6,"0")+".png",img_pos)
                        labels=np.append(labels,[('img_'+str(index).rjust(2,"0")+"_"+str(count).rjust(6,"0")+".png",
                                                        img_pos.shape[0],img_pos.shape[1],index,pos[0],pos[1],pos[0]+img_pos.shape[1],pos[1]+img_pos.shape[0])],0)

                        #debug print(labels)
                        

                        count = count +1
                        #cv2.imwrite()
                        #labels
                
end = time.time()
print("Time: "+ str(end - start))


np.savetxt(csv_name+".csv", labels, delimiter=",",fmt='%s')

#cv2.waitKey(1000)
cv2.destroyAllWindows()
#np.append(test,[images[1]],0)



# %%

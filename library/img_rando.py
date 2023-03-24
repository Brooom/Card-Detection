# Tutorial by https://medium.com/swlh/image-classification-for-playing-cards-26d660f3149e
#%%
import numpy as np
import cv2  
import sys
import glob
import random
import math

fix_seed = None

def def_fix_seed(_seed=None):
    """
    Load data from a text file, with missing values handled as specified.

    Each line past the first `skip_header` lines is split at the `delimiter`
    character, and characters following the `comments` character are discarded.

    Parameters
    ----------
    _seed : any
        If set to not None (default) ther is no global seed.
        If set to any thing overrydes any seed in this module  
    """
    global fix_seed 
    fix_seed = _seed
    return fix_seed

def img_zoom_rand(img,seed=None):
    """
    zooms a image randomly and recenters it,
     The scale fator will be in the range of 0.25 to 1

    Parameters
    ----------
    img : np.array  
        a numpy array (or comp) array aka image
    seed : random.seed comp, optional
        a seed for the ramdeom generater,
        if None and fix seed is has a vale (not None)
        fix_seed is used, by default None

    Returns
    -------
    np.array
        of the same type as and size as the imput array
    """

    if fix_seed != None:
        random.seed(fix_seed)
        #debug print(2)
    elif seed != None:
        random.seed(seed)
        #debug print(3)
    
    sc = random.uniform(1/4,1)

    #debug cv2.imshow("debug_zoom",cv2.resize(img, (0, 0), fx=sc, fy=sc))
    return cv2.resize(img, (0, 0), fx=sc, fy=sc)


def img_rot_rand(img,seed=None):
    """
    rotats a image randomly around its center it.

    Parameters
    ----------
    img : np.array  
        a numpy array (or comp) array aka image
    seed : random.seed comp, optional
        a seed for the ramdeom generater,
        if None and fix seed is has a vale (not None)
        fix_seed is used, by default None


    Returns
    -------
    np.array
        of the same type as and size as the imput array
    """

    if fix_seed != None:
        random.seed(fix_seed)
        #debug print("fix seed")
    elif seed != None:
        random.seed(seed)
        #debug print("seed")

    y, x= img.shape
    max = math.sqrt(x**2+y**2)
    cX, cY , cmax=  x/2, y/2, max/2


    M = cv2.getRotationMatrix2D((int(max/2),int(max/2)),  random.uniform(0,360),1.0)

    #rotated = cv2.warpAffine(img, M, (y,y))
    #cv2.imshow("Rotated by 45 Degrees", rotated)
    M_ = np.array(([[  1,   0, cmax-cX],
                    [  0,   1,  cmax-cY]]))
    #print(M*M_)
    #debug print( (int(max),int(max)))
    temp = cv2.warpAffine(img, M_, (int(max),int(max)))
    # cv2.imshow("temp",temp)
    # cv2.imshow("test",cv2.warpAffine( temp,M, (max,max)))
    return cv2.warpAffine( temp,M, (int(max),int(max)))


def img_3D_rand(img,seed=None):
    """
    performs a random 3D transformation with the image. (The image might be off center after transformation)

    Parameters
    ----------
    img : np.array  
        a numpy array (or comp) array aka image
    seed : random.seed comp, optional
        a seed for the ramdeom generater,
        if None and fix seed is has a vale (not None)
        fix_seed is used, by default None


    Returns
    -------
    np.array
        of the same type as and size as the imput array
    """

    if fix_seed != None:
        random.seed(fix_seed)
        #debugprint("fix seed")
    elif seed != None:
        random.seed(seed)
        ##print("seed")

    y, x= img.shape

    sc = 4
    rand_x = random.sample(range(int(x/sc)),4)
    rand_y = random.sample(range(int(y/sc)),4)

    pts1 = np.float32([ [0, 0], 
                        [0,y ],
                        [x,y],
                        [x,0 ]])

    pts2 = np.float32([[   rand_x[0],  rand_y[0]],
                            [  rand_x[1],y-rand_y[1]],
                            [x-rand_x[2],y-rand_y[2]], 
                           [x-rand_x[3],  rand_y[3]]])
    pts3 = np.float32([
        [ 44. ,  0.],
        [376., 180.],
        [ 32., 209.],
        [379. , 70.]])                   
    #print (pts2)
    M = cv2.getPerspectiveTransform(pts1,pts2)
    #cv2.imshow("test",cv2.warpPerspective(img, M, (x ,y)))
    return cv2.warpPerspective(img, M, (x ,y))

def img_pos_rand(img,dst,mask=None,seed=None):
    """
    Copys 'img' into 'dst' at a random position

    Parameters
    ----------
    img : np.array  
        a numpy array (or comp) array aka image
    dst : np.array  
        a numpy array (or comp) array aka image that need to be bigger or the same size as 'img'
    mask : np.array  
        a numpy array (or comp) array aka image with the same size as 'img' for masking ist coppy action into 'dst'
    
    seed : random.seed comp, optional
        a seed for the ramdeom generater,
        if None and fix seed is has a vale (not None)
        fix_seed is used, by default None

    Returns
    -------
    np.array
        of the same type as and size as the imput array
    """

    if fix_seed != None:
        random.seed(fix_seed)
        #debug print("fix seed")
    elif seed != None:
        random.seed(int(seed))
        #debug print("seed")
    
    _dst = np.copy(dst)
    
    if mask is None:
        ret, mask = cv2.threshold(img, 0, 255, 0)
    ret, mask = cv2.threshold(img, 0, 255, 0)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1) 
    y, x= img.shape
    dst_y, dst_x= dst.shape

    rand_x = random.randint(0,dst_x-x)
    rand_y = random.randint(0,dst_y-y)

    #debug print(img.shape,dst.shape,dst[rand_y:rand_y+y,rand_x:rand_x+x].shape,rand_x,rand_y)
    np.copyto(_dst[rand_y:rand_y+y,rand_x:rand_x+x],img,casting="unsafe",where=mask> 100)

    #debugcv2.imshow('replace', _dst)
    #debug cv2.waitKey(1000)
    return _dst,(rand_x,rand_y)

def scale_imput_img():
    """read all 'jpg' from the folder 'backgrounds' and save the back to the older as grascale with size 800*514

    Returns
    -------
    int
        0, to see i function has compleeted ( !no error reporting)
    """

    for file in glob.glob("background/*.jpg"):
        img = cv2.imread(file,0)
        sc = 2
        cv2.imwrite(file, cv2.resize(img, (400*sc, 257*sc)))


    return 0






#%%
def img_kill (img,delay = 5000 ):
    """desplay a 'img' with open cv in its one window and close it afert 'delay' or a pressed key

    Parameters
    ----------
    img : np.arra
        a image
    delay : int, optional
        time until windwo coloses automaticly, by default 5000
    """

    cv2.imshow("img_kill",img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

# testcode for funktios
if __name__ == "__main__":
    seed = (1,2)
    random.seed(seed)

    sys.path.append('D:/FHGR/05_HS_22/BiV_02/00_Projekt/Biv2')
    sys.path.append('E:\FHGR\Biv2')

    #%matplotlib inline

    #%% import images and cvs

    path = 'Cards'

    card_list_dtype = (int,str,str,ord,int)

    images = [cv2.imread(file,0) for file in glob.glob(path+"/*.png")]
    images = np.array(images)

    back = [cv2.imread(file,0) for file in glob.glob("background/*.jpg")]
    back_images = np.array(back)

    #%% remove black blank
    print(images.shape)
    ##images = images[1:]
    print(images.shape)
    #%% 
    card_list = np.loadtxt('card_list.csv',delimiter=";", dtype=str)[1:]
    for col in range(card_list.shape[1]):
        print(card_list_dtype[col])

    #%%zoom test


    img = images[1]
    rows, cols= img.shape


    for sc in range(1,10):
        print(sc)   
        pts1 = np.float32([[0, 0],
                        [cols,rows], 
                        [0,rows ]])
        pts2 = np.float32([[0, 0],
                        [cols/sc, rows/sc], 
                        [0,rows/sc ]])
        M = cv2.getAffineTransform(pts1, pts2)
        cv2.imshow("test",cv2.warpAffine(img, M, (cols ,rows)))
        cv2.imshow("test 2 ",cv2.resize(img, [cols,rows], 0.5, 0.5))
        cv2.waitKey(500)

    cv2.destroyAllWindows()

    #%% test trans
    img = images[1]
    rows, cols= img.shape

    sc = np.sqrt(2) #scale 
    
    pts1 = np.float32([[0, 0],
                    [cols,rows], 
                    [0,rows ]])

    x = 20
    y = 50
    pts2 = np.float32([[x, y], #pos
                    [x + cols/sc, y+rows/sc], #sclae
                    [x,y+rows/sc ]]) #trans
    
    M = cv2.getAffineTransform(pts1, pts2)
    cv2.imshow("test",cv2.warpAffine(img, M, (cols ,rows)))

    # saninty check if img is in frame
    if cv2.calcHist([img], [0], None, [256], [0, 256])[240:].sum():
        print("not correctly implementeds use scal and so on")
    cv2.waitKey(5000)

    cv2.destroyAllWindows()

    #%% test 3d rot

    cv2.imshow("test",img)
    cv2.waitKey(0)
    for i in range(100):
        random.seed(i)
        img = images[1]
        rows, cols= img.shape
        sc = 4
        rand_x = random.sample(range(int(cols/sc)),4)
        rand_y = random.sample(range(int(rows/sc)),4)

        #rand_x = rand_y = [50,50,50,50]

        pts1 = np.float32([ [0, 0], 
                            [0,rows ],
                            [cols,rows],
                            [cols,0 ]])

        pts2 = np.float32([[rand_x[0]       , rand_y[0]],
                            [rand_x[1]      ,rows-rand_y[1]],
                            [cols-rand_x[2] ,rows-rand_y[2]], 
                            [cols-rand_x[3] ,rand_y[3]]])
        pts3 = np.float32([
        [ 44. ,  0.],
        [376., 180.],
        [ 32., 209.],
        [379. , 70.]])                   
        #print (pts2)
        M = cv2.getPerspectiveTransform(pts1,pts2)
        cv2.imshow("test",cv2.warpPerspective(img, M, (cols ,rows)))
        cv2.waitKey(50)

    cv2.destroyAllWindows()

    #%%
    for img in images:
        #cv2.rectangle(img,[10,10],[247,390],255,-1)
        cv2.imshow("test",img)
        cv2.waitKey(0)

    cv2.waitKey(1000)
    cv2.destroyAllWindows()




















cv2.destroyAllWindows()



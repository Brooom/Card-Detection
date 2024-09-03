# Tutorial by https://medium.com/swlh/image-classification-for-playing-cards-26d660f3149e
import glob
import random
import math
import numpy as np
import cv2

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

def img_zoom_rand(img,seed=None, minZoom=1/4, maxZoom=1):
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

    sc = random.uniform(minZoom, maxZoom)
    # print("Zoomfactor: "+ str(sc))
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

    y, x, c= img.shape
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

    y, x, c= img.shape

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
        #Create mask
        ret, mask = cv2.threshold(img, 0, 255, 0, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1) 

    #Generate random pos
    y, x, c= img.shape
    dst_y, dst_x, dst_c= dst.shape
    rand_x = random.randint(0,dst_x-x)
    rand_y = random.randint(0,dst_y-y)

    #Define how transparent the card should be
    transparency=random.uniform(0,0.4)

    #Generate a big mask where the small mask is placed at the random position
    alpha=np.zeros_like(_dst, dtype='uint8')
    np.copyto(alpha[rand_y:rand_y+y,rand_x:rand_x+x],mask,casting="unsafe")

            #np.copyto(_dst[rand_y:rand_y+y,rand_x:rand_x+x],img,casting="unsafe",where=mask> 100)

    #Place the card at the random posion in the mask. This is a black image with the card placed at the random position
    forground=np.zeros_like(_dst, dtype='uint8')
    np.copyto(forground[rand_y:rand_y+y,rand_x:rand_x+x],img,casting="unsafe", where=mask> 100)

    # Convert uint8 to float
    forground = forground.astype(float)
    _dst = _dst.astype(float)

    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha.astype(float)/255

    # Multiply the foreground with the alpha matte. This makes the forground image a little bit darker. Now the card will not be to light in the final image
    forground = cv2.multiply(alpha-transparency, forground)

    # The mask (alpha) is inverted. Now we have the background image with a black spot where the card is.
    # Because we want the card to be transparent, we add a scalar transparency. This results in the black area no longer being completely black.
    # Because of the transparency factor, we can see the background in the black spot a little bit.
    # The factor calculated is normalised to 1 so the final image is not too bright
    _dst = cv2.multiply((1-alpha+transparency)/(1+transparency), _dst)

    # Add the masked foreground and background are added.
    # Because the forgound image is a little bit less bright and the background is a little bit visible at the black spot, we now have a transparent card
    outImage = cv2.add(forground, _dst)

    # Display image
    """
    cv2.imshow("forground", forground/255)
    cv2.imshow("_dst", _dst/255)
    cv2.imshow("outImg", outImage/255)
    cv2.waitKey(0)
    """

    #debugcv2.imshow('replace', _dst)
    #debug cv2.waitKey(1000)
    return outImage/255,(rand_x,rand_y)

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


def img_blure(img, kernel_size=3):
    """blures a image

    Parameters
    ----------
    img : np.array
        a image
    delay : int, optional
        How big the kernel is, by default 3
    """
    # Creating the kernel(2d convolution matrix)
    kernel1 = np.ones((kernel_size, kernel_size), np.float32)*1/(kernel_size*kernel_size)

    # Applying the filter2D() function
    blured_img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel1)
    return blured_img


# testcode for functinos
if __name__ == "__main__":
    path_cards = '0_cards_images'
    path_background = '0_background_images'

    card=cv2.imread(glob.glob(path_cards+"/*.png")[0])
    background=cv2.imread(glob.glob(path_background+"/*.jpg")[0])
    card[card==0]=1
    while True:
        res=img_pos_rand(img_rot_rand(card), background)
        cv2.imshow("test", res[0])
        cv2.waitKey(0)
    cv2.destroyAllWindows()

import numpy as np
def darkening(list_of_images,factor):
    return list_of_images/factor
    
def shift_left(list_of_images, pixels):

    for i in range(len(list_of_images)):
        list_of_images[i] = np.roll(list_of_images[i], -pixels, axis=0)
    return list_of_images

def shift_right(list_of_images, pixels):

    for i in range(len(list_of_images)):
        list_of_images[i] = np.roll(list_of_images[i], pixels, axis=0)
    return list_of_images

def move_up(list_of_images, pixels):

    for i in range(len(list_of_images)):
        list_of_images[i] = np.roll(list_of_images[i], -pixels, axis=1)
    return list_of_images

def move_down(list_of_images, pixels):

    for i in range(len(list_of_images)):
        list_of_images[i] = np.roll(list_of_images[i], pixels, axis=1)
    return list_of_images

def flip_horizontal_inplace(list_of_images):

    for i in range(len(list_of_images)):
        list_of_images[i] = np.fliplr(list_of_images[i])
    return list_of_images

def flip_vertical_inplace(list_of_images):

    for i in range(len(list_of_images)):
        list_of_images[i] = np.flipud(list_of_images[i])
    return list_of_images
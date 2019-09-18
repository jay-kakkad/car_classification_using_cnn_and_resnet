# -*- coding: utf-8 -*
"""
Created on Tue Dec 19 10:59:50 2017

@author: jpkak
"""

import glob
import matplotlib.image as img
import numpy as np
from PIL import Image



"""
class Load_image_into_numpy:
        def __init__(self,filename):
            self.im=img.imread(filename)
            print("Filename:",filename,"\t\tImage_Shape:",self.im.shape)"""
            
def load_numpy_as_array(image):
    (img_width,img_height) = image.size
    return np.array(image.getdata()).reshape((img_height,img_width,3)).astype(np.float32)    

image_np =[]

	
    image = Image.open(image_path)
    image_np.append(load_numpy_as_array(image))
    print(image_path)
  
            
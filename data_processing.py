# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 12:17:14 2017

@author: jpkak
"""

from keras.preprocessing.image import ImageDataGenerator

def combine_sets(shear_range,zoom_range,set_dir,horizontal_flip=False,vertical_flip=False,batch_size=0,
                 target_size,class_mode='binary',target_size):
    img_h,img_w=target_size
    image_datagen = ImageDataGenerator(rescale = 1./255,shear_range=shear_range,zoom_range=zoom_range,
                                       horizontal_flip=horizontal_flip,vertical_flip=vertical_flip,
                                       target_size=(img_h,img_w),batch_size=batch_size)
    image_set=train_datagen.flow_from_directory(set_dir,target_size=(img_h,img_w),
                                                batch_size=32,class_mode=class_mode)
    
    return image_set
    
    
    
    
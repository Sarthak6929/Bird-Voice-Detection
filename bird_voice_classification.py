#!/usr/bin/env python
# coding: utf-8

# In[5]:


import cv2 # importing OpenCV 
import time
from keras.models import Sequential # Sequential layers for buildinig Neural Networks
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten # Importing different layers of CNN from keras library 
#MaxPooling2D for taking out the max value from the kernal taken in 2D for training, Dense used for the layers used for making Neural Network,
#flatten to convert 1d to 2d.
from keras.optimizers import Adam # Keeps optimizing the loss, Adam is an optimizer used to reach the global minima from the loss function by using gradient descent 
from keras.preprocessing.image import ImageDataGenerator # image generator helps converting all the images to json files
from keras.callbacks import EarlyStopping


# In[6]:


train_data_gen = ImageDataGenerator(
    rescale=1./255)

validation_data_gen = ImageDataGenerator(rescale=1./255) # the data is necessary to get in this range so that it is easier for the machine to create graphs.
# path for the training and validation data directories


# In[7]:


train_generator = train_data_gen.flow_from_directory(
        'train', # folder name
        target_size=(100, 100), #image size 100X100
        batch_size=64, #alag alag batch of 64 will be made. 
        color_mode="grayscale", #Image colour to grey so that it's easy to apply kernel, understand max pool, and to take out pixels from the image.
        class_mode='categorical') #to change pixels into numbers, machine can't understand pixels. 


# In[8]:


validation_generator = validation_data_gen.flow_from_directory(
        'test',
        target_size=(100, 100),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')


# In[9]:


model = Sequential() # calling Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(100, 100, 1))) # adding 32 neurons of conv2D, kernel size 3X3 and activation function 'relu'
#input shape 100X100X1 in 3D shapes 
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu')) 
model.add(MaxPooling2D(pool_size=(2, 2))) # applying max pool, get the maximum value after passing all the images through th kernel. max pool will be in 2d
model.add(Dropout(0.25)) # testing value to be near 0.25% then it's ok

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten()) # converting all the data into 1d
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5)) #testing accuracy kept to be 0.5%
model.add(Dense(7, activation='softmax')) # number of folder 7 


# In[10]:


cv2.ocl.setUseOpenCL(False) #passing it into cv2 , used openCl 
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy']) # complied the model, learning rate = 0.0001, decay =  not such special use, Optimizer Adam is also called to reach the global minima 


# In[ ]:


start_time = time.time()
model_info = model.fit_generator( 
        train_generator,
        steps_per_epoch=6, # make group of 6
        epochs=85, # read 85 times
        validation_data=validation_generator,
        validation_steps=6, # test in group of 6
        callbacks=[early_stop])

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time: {:.2f} hour".format(elapsed_time/3600))


# In[ ]:


model.save("bird_voice.h5")


# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 16:25:12 2018

@author: GITESH
"""

from skimage import io,color
from skimage.transform import resize
import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense,Dropout,Concatenate, Activation,merge , ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import scipy.misc
from keras.optimizers import *
from matplotlib import pyplot as plt
from numpy import array
from matplotlib.pyplot import imshow
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
from keras.preprocessing.image import ImageDataGenerator
import json
bn='bn_layer_'
conv='conv_layer_'
fc= 'fc_layer_'
k=32
def save_history(history,file):
    with open(file, 'w') as f:
        json.dump(history, f)
    '''
    data = dict()
    with open('mydatafile') as f:
        data = json.load(f)
    '''
def bottleneck_composite(l,layer):
    # bottleneck layer
    X=l
    if type(l) is list:
        if(len(l)==1):
            X=l[0]
        else:
            X=Concatenate(axis=-1)(l)
            #X= merge(l, mode='concat', concat_axis=-1)

    X = BatchNormalization(axis = 3, name = bn + str(layer))(X)
    X = Activation('relu')(X)
    X = Conv2D(4*k, (1, 1), strides = (1, 1),padding='same', name = conv + str(layer), kernel_initializer = glorot_uniform(seed=0))(X)
    X = Dropout(0.8)(X)
    # Composite layer
    layer=layer+1
    X = BatchNormalization(axis = 3, name = bn +  str(layer))(X)
    X = Activation('relu')(X)
    X = Conv2D(k, (3, 3), strides = (1, 1),padding='same', name = conv +  str(layer), kernel_initializer = glorot_uniform(seed=0))(X)
    X = Dropout(0.8)(X)
    return X
    
    
layer=0    
def chexnet(classes=14,input_shape=(224,224,3)):
    X_input = Input(input_shape)
    layer=0
    layer=layer+1
    X = ZeroPadding2D((3, 3))(X_input)
    X = BatchNormalization(axis = 3, name = bn + str(layer))(X)
    X = Activation('relu')(X)
    X = Conv2D(2*k, (7, 7), strides = (2, 2), name = conv + str(layer), kernel_initializer = glorot_uniform(seed=0))(X)
    X = Dropout(0.8)(X)
    print(X.shape)
    X = ZeroPadding2D((1, 1))(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)
    print(X.shape)
    #Dense Block = 1
    layer=layer+1
    X=bottleneck_composite(X,layer)
    l=[]
    l.append(X)
    for i in range(0,5):
        layer=layer+2
        X=bottleneck_composite(l,layer)
        l.append(X)
    print(X.shape)
    # Transition layer = 1   
    layer=layer+2
    X = BatchNormalization(axis = 3, name = bn +  str(layer))(X)
    X = Activation('relu')(X)
    X = Conv2D(k, (1, 1), strides = (1, 1),padding ='same', name = conv +  str(layer), kernel_initializer = glorot_uniform(seed=0))(X)
    X = Dropout(0.8)(X)
    X = AveragePooling2D((2, 2), strides=(2, 2))(X)  
    print(X.shape)
    
    #Dense Block = 2
    layer=layer+1
    X=bottleneck_composite(X,layer)
    l=[]
    l.append(X)
    for i in range(0,11):
        layer=layer+2
        X=bottleneck_composite(l,layer)
        l.append(X)
    
    print(X.shape)
    # Transition layer = 2
    layer=layer+2
    X = BatchNormalization(axis = 3, name = bn +  str(layer))(X)
    X = Activation('relu')(X)
    X = Conv2D(k, (1, 1), strides = (1, 1),padding ='same', name = conv +  str(layer), kernel_initializer = glorot_uniform(seed=0))(X)
    X = Dropout(0.8)(X)
    X = AveragePooling2D((2, 2), strides=(2, 2))(X)  
    print(X.shape)
    #Dense Block = 3
    layer=layer+1
    X=bottleneck_composite(X,layer)
    l=[]
    l.append(X)
    for i in range(0,23):
        layer=layer+2
        X=bottleneck_composite(l,layer)
        l.append(X)
    print(X.shape)
    # Transition layer = 3
    layer=layer+2
    X = BatchNormalization(axis = 3, name = bn +  str(layer))(X)
    X = Activation('relu')(X)
    X = Conv2D(k, (1, 1), strides = (1, 1),padding ='same', name = conv +  str(layer), kernel_initializer = glorot_uniform(seed=0))(X)
    X = Dropout(0.8)(X)
    X = AveragePooling2D((2, 2), strides=(2, 2))(X)  
    print(X.shape)
    #Dense Block = 4
    layer=layer+1
    X=bottleneck_composite(X,layer)
    l=[]
    l.append(X)
    for i in range(0,15):
        layer=layer+2
        X=bottleneck_composite(l,layer)
        l.append(X)
    print(X.shape)
    layer=layer+2
    print(X.shape)
    X=  GlobalAveragePooling2D()(X)
    print(X.shape)
    # fully connected layer
    #X = Flatten()(X)
    X = Dense(classes, activation='softmax', name=  fc  +  str(layer), kernel_initializer = glorot_uniform(seed=0))(X)
    print(X.shape)
    model = Model(inputs = X_input, outputs = X, name="DenseNet121")
	
    return model

adam=Adam(lr=0.001)
model = chexnet(classes = 4,input_shape = (224,224,3))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#model.summary()
train_datagen = ImageDataGenerator( rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen=ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        'Train',
        target_size=(224,224),
        batch_size=32,
        class_mode='categorical')
validation_generator = val_datagen.flow_from_directory(
        'Validation',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')
test_generator=test_datagen.flow_from_directory(
        'Test',
        target_size=(224,224), 
        batch_size=32,
        class_mode='categorical')
print(train_generator.class_indices)
print(test_generator.class_indices)
print(validation_generator.class_indices)
'''
model=load_model('my_densenet')
adam=Adam(lr=0.001)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
'''
history=model.fit_generator(train_generator, epochs =25,steps_per_epoch=2153,validation_data=validation_generator, validation_steps=175)
model.save('my_densenet25_with_dropout_new_data')
    #print(model_files[i])
    #print(history.history.keys())  
save_history(history.history,'history_densenet25_with_dropout_new_data')
    #print(history_files[i])


preds = model.evaluate_generator(train_generator, steps=2153)
print ("train Loss = " + str(preds[0]))
print ("train Accuracy = " + str(preds[1]))
preds = model.evaluate_generator(validation_generator, steps=175)
print ("validation Loss = " + str(preds[0]))
print ("validation Accuracy = " + str(preds[1]))
preds = model.evaluate_generator(test_generator, steps=175)
print ("test Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
#model=load_model(model_files[i])
    #print(model_files[i]) 
	
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()          
###

print("DONE")
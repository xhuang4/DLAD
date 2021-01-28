'''
This file contains code to run a VGG16 transfer learning classifier trained by MRI T1 or Log Jacobian
AD/CN images processed by the ANTs pipeline (Avants et al., 2010). Model output is the features of the last network layer. 
Trained model is used to predict MCI images.
'''

import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Dense
from keras import applications

# image dimensions and parameters
img_width, img_height = 150, 150
nb_train_samples = 553
nb_validation_samples = 139
epochs = 100
batch_size = 7


# image path
train_data_dir = './training_data'
validation_data_dir = './validation_data'
MCI_data_dir = './MCIs'

#load images
datagen = ImageDataGenerator(rescale=1. / 255)
generator_train = datagen.flow_from_directory(
train_data_dir,
target_size=(img_width, img_height),
batch_size=batch_size,
class_mode=None,
shuffle=False)
train_labels = np.array(generator_train.classes)

generator_valid = datagen.flow_from_directory(
validation_data_dir,
target_size=(img_width, img_height),
class_mode=None,
batch_size = 1,
shuffle=False)
validation_labels = np.array(generator_valid.classes)

generator_MCI = datagen.flow_from_directory(
    MCI_data_dir,
    target_size=(img_width, img_height),
    batch_size=1,
    class_mode=None,
    shuffle=False)


# Extract features from pretrained VGG16 network
modelVGG = applications.VGG16(include_top=False, weights='imagenet')
bottleneck_features_train = modelVGG.predict_generator(generator_train, nb_train_samples // batch_size)
bottleneck_features_validation = modelVGG.predict_generator(
generator_valid, nb_validation_samples)
bottleneck_features_MCI = modelVGG.predict_generator(generator_MCI,steps = len(generator_MCI))


# Build and train the last layers 
model = Sequential()
model.add(Flatten(input_shape=bottleneck_features_train.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['accuracy'])
saveBestModel=keras.callbacks.ModelCheckpoint('./VGG16model.hdf5', monitor='val_acc', mode='max',verbose=1, save_best_only=True)
model.fit(bottleneck_features_train, train_labels,epochs=epochs,validation_data=(bottleneck_features_validation, validation_labels),shuffle = True, callbacks=[saveBestModel])  

#Extract features from the last layers
model_trained = load_model('./VGG16model.hdf5')
outputs = model_trained.layers[2].output
newmodel = Model(inputs=model_trained.inputs,outputs=outputs)
newmodel.save('VGG16model_outputFeatures.hdf5')

#Model prediction for MCI images
predict_prob=newmodel.predict(bottleneck_features_MCI)
np.savetxt('MCI_outputFeatures.txt',predict_prob,fmt='%.4f')

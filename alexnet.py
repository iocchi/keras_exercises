# https://www.learnopencv.com/understanding-alexnet/

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, Flatten,\
 Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np

np.random.seed(20181205)

import tflearn.datasets.oxflower17 as oxflower17

def load_data():
    Xtrain, Ytrain = oxflower17.load_data(one_hot=True)
    
    input_shape = (Xtrain.shape[1], Xtrain.shape[2], Xtrain.shape[3])     # (224,224,3)
    num_classes = Ytrain.shape[1]  # 17
    print("Training input %s" %str(Xtrain.shape))
    print("Training output %s" %str(Ytrain.shape))
    #print("Test input %s" %str(Xtest.shape))
    #print("Test output %s" %str(Ytest.shape))
    print("Input shape: %s" %str(input_shape))
    print("Number of classes: %d" %num_classes)

    return [Xtrain,Ytrain,input_shape,num_classes] 


filename = 'alexnet_oxflower17.h5'

def savemodel(model):  
    model.save(filename)
    print("\nModel saved successfully on file %s\n" %filename)


def loadmodel():
    try:
        model = load_model(filename)
        print("\nModel loaded successfully from file %s\n" %filename)
    except OSError:
        print("\nModel file %s not found!!!\n" %filename)
        model = None
    return model


def AlexNet(input_shape, num_classes):
    # (3) Create a sequential model
    model = Sequential()

    # 1st Convolutional Layer 
    model.add(Conv2D(filters=96, input_shape=input_shape, kernel_size=(11,11),\
     strides=(4,4), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation before passing it to the next layer
    model.add(BatchNormalization())

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Passing it to a dense layer
    model.add(Flatten())
    # 1st Dense Layer
    model.add(Dense(4096, input_shape=(224*224*3,)))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 2nd Dense Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # 3rd Dense Layer
    model.add(Dense(1000))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Output Layer
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.summary()

    # (4) Compile
    model.compile(loss='categorical_crossentropy', optimizer='adam',\
    metrics=['accuracy'])

    return model
 

    

### main ###
if __name__ == "__main__":

    # Get Data
    [Xtrain,Ytrain,input_shape,num_classes] = load_data()
    
    # Load or create model
    model = loadmodel()
    if model==None:
        model = AlexNet(input_shape, num_classes)

    # Train
    try:
        model.fit(Xtrain, Ytrain, batch_size=64, epochs=1, verbose=1, \
        validation_split=0.2, shuffle=True)
    except KeyboardInterrupt:
        pass
      
    # Save the model
    savemodel(model)

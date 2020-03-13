import os, sys
import numpy as np

import keras
from keras import models, layers, backend
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten,\
 Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import to_categorical

models_dir = 'models'

trainingset = 'datasets/ARGOS-sc5-2013/train'
testset = 'datasets/ARGOS-sc5-2013/test'

classes = [ 'Alilaguna',\
          'Ambulanza',\
          'Barchino',\
          'Cacciapesca',\
          'Caorlina',\
          'Gondola',\
          'Lancia',\
          'Motobarca',\
          'Motopontonerettangolare',\
          'MotoscafoACTV',\
          'Mototopo',\
          'Patanella',\
          'Polizia',\
          'Raccoltarifiuti',\
          'Sandoloaremi',\
          'Sanpierota',\
          'Topa',\
          'VaporettoACTV',\
          'VigilidelFuoco',\
          'Acqua'\
        ]

groups = [ ['Alilaguna'],\
           ['Ambulanza'],\
           ['Barchino'],\
           ['Cacciapesca'],\
           ['Gondola'],\
           ['Lancia'],\
           ['Motobarca'],\
           ['Motopontonerettangolare'],\
           ['Mototopo'],\
           ['Patanella'],\
           ['Polizia'],\
           ['Raccoltarifiuti'],\
           ['Sandoloaremi','Caorlina'],\
           ['Sanpierota','Topa'],\
           ['VaporettoACTV','MotoscafoACTV'],\
           ['VigilidelFuoco'],\
           ['Acqua']\
         ]

groups_enabled = False
loadclasses = False
whichgroup = None

batch_size = 32

def load_image(filename):
    img = load_img(filename)
    img = img.resize((224,118))
    x = img_to_array(img)  # this is a Numpy array with shape (3, width, height)
    x /= 255.0  # normalization   
    #print('image shape: %s' %str(x.shape))
    return x


def findgroup(strclass):
    global whichgroup
    if whichgroup==None:
        whichgroup = {}
        for cl in classes:
            igr = 0
            for gr in groups:
                if cl in gr:
                    whichgroup[cl] = igr
                igr+=1

    strclass = strclass.replace(":","")
    strclass = strclass.replace(" ","")
    strclass = strclass.strip()
    try:
        igr = whichgroup[strclass]
    except KeyError:
        igr = -1 
    return igr

def load_data():

    train_datagen = ImageDataGenerator(
        rescale = 1. / 255,\
        zoom_range=0.1,\
        rotation_range=10,\
        width_shift_range=0.1,\
        height_shift_range=0.1,\
        horizontal_flip=True,\
        vertical_flip=False)

    train_generator = train_datagen.flow_from_directory(
        directory=trainingset,
        target_size=(118, 224),
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True
    )

    test_datagen = ImageDataGenerator(
        rescale = 1. / 255)

    test_generator = test_datagen.flow_from_directory(
        directory=testset,
        target_size=(118, 224),
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True
    )

    num_samples = train_generator.n
    num_classes = train_generator.num_classes
    input_shape = train_generator.image_shape

    classnames = [k for k,v in train_generator.class_indices.items()]

    print("Image input %s" %str(input_shape))
    print("Classes: %r" %classnames)
    print('Loaded %d training samples from %d classes.' \
          %(num_samples,num_classes))
    print('Loaded %d test samples from %d classes.' \
          %(test_generator.n,test_generator.num_classes))

    return input_shape,num_classes,train_generator,test_generator
 

def transform_test_data():
    gtfile = 'ground_truth.txt'
    cc = {}
    cn = 0
    fgt = open(os.path.join(testset,gtfile), 'r')
    for l in fgt:
        v = l.split(';') # filename;class
        filename = v[0].strip()
        classname = v[1].strip()

        if not classname in cc.keys():
            cc[classname]=0
        cc[classname]+=1
        if classname[0]!='_':
            cn += 1

        print('move %s to %s' %(filename,classname))
        st = None
        try:
            st = os.stat(classname)
        except OSError:
            pass
        if st==None:
            os.mkdir(os.path.join(testset,classname))
        f1 = os.path.join(testset,filename)
        f2 = os.path.join(os.path.join(testset,classname),filename)
        os.rename(f1,f2)

    fgt.close()
    print(cc)
    print('Totale %d' %cn)

def _OLD_load_data():
    Xtrain = []
    Ytrain = []
    #mem = 0

    if groups_enabled:
        grid=0
        for gr in groups:
            print('Group %d' %grid)
            for cl in gr:
                cdir = os.path.join(trainingset,cl)
                print('  %s' %cdir)
                try:
                    ff = os.listdir(cdir)    
                except:
                    print("  Cannot find %s" %cdir)
                    ff = []
                for f in ff:
                    fn = os.path.join(cdir,f)
                    x = load_image(fn)
                    #mem += x.shape[0]*x.shape[1]*x.shape[2]
                    #print("Memory used: %d" %mem)
                    y = grid
                    y = to_categorical(y,len(groups))
                    #print("%s" %str(y))
                    Xtrain.append(x)
                    Ytrain.append(y)
            grid += 1

    elif loadclasses:
        for cl in classes:
            cdir = os.path.join(trainingset,cl)
            print(cdir)
            try:
                ff = os.listdir(cdir)    
            except:
                print("Cannot find %s" %cdir)
                ff = []
            for f in ff:
                fn = os.path.join(cdir,f)
                x = load_image(fn)
                #mem += x.shape[0]*x.shape[1]*x.shape[2]
                #print("Memory used: %d" %mem)
                y = classes.index(cl)
                y = to_categorical(y,len(classes))
                #print("%s" %str(y))
                Xtrain.append(x)
                Ytrain.append(y)

    Xtrain = np.array(Xtrain)
    Ytrain = np.array(Ytrain)

    print("Training input %s" %str(Xtrain.shape))
    print("Training output %s" %str(Ytrain.shape))
    
    if Xtrain.shape[0]==0:
        input_shape=(118, 224, 3)
        num_classes=20
    else:
        input_shape = (Xtrain.shape[1], Xtrain.shape[2], Xtrain.shape[3])
        num_classes = Ytrain.shape[1]
    #print("Test input %s" %str(Xtest.shape))
    #print("Test output %s" %str(Ytest.shape))
    print("Input shape: %s" %str(input_shape))
    print("Number of classes: %d" %num_classes)


    # Load test data
    print('Loading test data')
    Xtest = []
    Ytest = []

    gtfile = os.path.join(testset,'ground_truth.txt')
    fgt = open(gtfile, 'r')
    for l in fgt:
        v = l.split(';') # filename;class
        igr = findgroup(v[1].strip())
        if (igr>=0):
            fn = os.path.join(testset,v[0].strip())
            x = load_image(fn)
            y = to_categorical(igr,num_classes)
            Xtest.append(x)
            Ytest.append(y)  
    fgt.close()

    Xtest = np.array(Xtest)
    Ytest = np.array(Ytest)

    print("Test input %s" %str(Xtest.shape))
    print("Test output %s" %str(Ytest.shape))
 
    return [Xtrain,Ytrain,Xtest,Ytest,input_shape,num_classes] 


def AlexNet(input_shape, num_classes):

    model = Sequential()

    # C1 Convolutional Layer 
    model.add(Conv2D(filters=96, input_shape=input_shape, kernel_size=(11,11),\
                     strides=(2,4), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation before passing it to the next layer
    model.add(BatchNormalization())

    # C2 Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # C3 Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # C4 Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # C5 Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Flatten
    model.add(Flatten())

    regl2 = 0.0001

    # D1 Dense Layer
    model.add(Dense(4096, input_shape=(input_shape[0]*input_shape[1]*input_shape[2],), kernel_regularizer=regularizers.l2(regl2)))
    model.add(Activation('relu'))
    # Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # D2 Dense Layer
    model.add(Dense(4096, kernel_regularizer=regularizers.l2(regl2)))
    model.add(Activation('relu'))
    # Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # D3 Dense Layer
    model.add(Dense(1000,kernel_regularizer=regularizers.l2(regl2)))
    model.add(Activation('relu'))
    # Dropout
    model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Output Layer
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # Compile

    adam = optimizers.Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model
 


 

def savemodel(model,problem):
    if problem.endswith('.h5'):
        filename = problem
    else:
        filename = os.path.join(models_dir, '%s.h5' %problem)
    model.save(filename)
    #W = model.get_weights()
    #print(W)
    #np.savez(filename, weights = W)
    print("\nModel saved successfully on file %s\n" %filename)

    
def loadmodel(problem):
    if problem.endswith('.h5'):
        filename = problem
    else:
        filename = os.path.join(models_dir, '%s.h5' %problem)
    try:
        model = models.load_model(filename)
        print("\nModel loaded successfully from file %s\n" %filename)
    except OSError:    
        print("\nModel file %s not found!!!\n" %filename)
        model = None
    return model
  

### main ###
if __name__ == "__main__":

    problem = 'ARGOS-sc5-2013_20classes_alexnet'

    #np.random.seed(20181205)

    # Get Data
    input_shape,num_classes,train_generator,test_generator = load_data()

    # Load or create model
    model = loadmodel(problem)
    if model==None:
        model = AlexNet(input_shape, num_classes)

    # Summary
    model.summary()

    # Train
    try:
        steps_per_epoch=train_generator.n//train_generator.batch_size
        val_steps=test_generator.n//test_generator.batch_size

        model.fit_generator(train_generator, epochs=100, verbose=1, \
                    steps_per_epoch=steps_per_epoch,\
                    validation_data=test_generator,\
                    validation_steps=val_steps)
    except KeyboardInterrupt:
        pass
      
    # Save the model
    savemodel(model,problem)

    print('Evaluation')
    loss, acc = model.evaluate_generator(test_generator,verbose=1,steps=val_steps)
    print('Test loss: %f' %loss)
    print('Test accuracy: %f' %acc)


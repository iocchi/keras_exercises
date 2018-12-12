# adapted from https://www.dlology.com/blog/how-to-do-unsupervised-clustering-with-keras/

import sys, os, datetime
import numpy as np
import argparse

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, classification_report
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn import svm

import keras
from keras import models, layers, backend
from keras.models import Model
from keras.layers import Input,Dense, Activation, Dropout, Flatten,\
    Conv2D, MaxPooling2D, AveragePooling2D
from keras.initializers import VarianceScaling
from keras.optimizers import SGD, Adam

models_dir = 'models'

def load_data():
    # load data
    (Xtrain,Ytrain), (Xtest, Ytest) = keras.datasets.mnist.load_data()
    # get information
    ninput = Xtrain.shape[0]
    ntest = Xtest.shape[0]
    num_classes = max(Ytrain) + 1
    
    # reshape to 1D vector
    Xtrain = Xtrain.reshape((Xtrain.shape[0], -1))
    Xtest = Xtest.reshape((Xtest.shape[0], -1))

    input_shape = (Xtrain.shape[0], 1)

    # normalize input to [0,1]
    Xtrain = Xtrain / 255.0
    Xtest = Xtest / 255.0

    print("Training input %s" %str(Xtrain.shape))
    print("Training output %s" %str(Ytrain.shape))
    print("Input shape: %s" %str(input_shape))
    print("Number of classes: %d" %num_classes)
    
    return [Xtrain,Ytrain,Xtest,Ytest,input_shape,num_classes]


    
def AutoEncoder(dims, act='relu', init='glorot_uniform'):
    
    print('\nAutoencoder')
        
    n_stacks = len(dims) - 1
    # input
    input_img = Input(shape=(dims[0],), name='input')
    x = input_img
    # internal layers in encoder
    for i in range(n_stacks-1):
        x = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(x)
        # hidden layer
    encoded = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(x)  # hidden layer, features are extracted from here
    
    x = encoded
    # internal layers in decoder\n",
    for i in range(n_stacks-1, 0, -1):
        x = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(x)
    
    # output\n",
    x = Dense(dims[0], kernel_initializer=init, name='decoder_0')(x)
    decoded = x

    model_ae = Model(inputs=input_img, outputs=decoded, name='AE')
    model_ae.compile(optimizer='adam', loss='mse')
        
    return model_ae #, model_enc


#nmi = normalized_mutual_info_score
#ari = adjusted_rand_score


def evaluation(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    
def savemodel(model,problem):
    if problem.endswith('.h5'):
        filename = problem
    else:
        filename = os.path.join(models_dir, '%s.h5' %problem)
    model.save(filename)
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

def loadskmodel(problem):
    if problem.endswith('.pkl'):
        filename = problem
    else:
        filename = os.path.join(models_dir, '%s.pkl' %problem)
    try:
        kmeans = joblib.load(filename)
        print("\nModel loaded successfully from file %s\n" %filename)
    except OSError:    
        print("\nModel file %s not found!!!\n" %filename)
        model = None
    return model

    
### main ###
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Keras LeNet example')
    parser.add_argument('-seed', type=int, help='random seed', default=0)
    args = parser.parse_args()

    # Load mnist dataset
    [Xtrain,Ytrain,Xtest,Ytest,input_shape,num_classes] = load_data()
        
    # Set random seed
    if args.seed==0:
        dt = datetime.datetime.now()
        rs = int(dt.strftime("%y%m%d%H%M"))
    else:
        rs = args.seed
    np.random.seed(rs)
    
    print("\nRandom seed %d" %rs)
    
    # PCA model
    pca = PCA(n_components=100)
    
    print("\nPCA Training ...")
    pca.fit(Xtrain)
    
    pca.fit(Xtrain)
    #means = pca.means_   #put this into a .npy file
    #components = pca.components_
    
    Xpca = pca.transform(Xtrain)

    if False:
        # equivalent to  pca.transform(Xtrain)
        from sklearn.utils.extmath import fast_dot
        td = Xtrain - means
        Xpca = fast_dot(td, components.T)

    if False:
        kmeans = KMeans(n_clusters=num_classes, n_init=1, n_jobs=2, verbose=0)

        print("\nK-means Training ...")
        # Train
        try:
            kmeans.fit(Xpca)
        except KeyboardInterrupt:
            pass

        print("\n\nEvaluation ...")
        # Evaluate the K-Means clustering accuracy
        y_pred_kmeans = kmeans.predict(Xpca)
        
        acc = evaluation(Ytrain, y_pred_kmeans)
        print("Accuracy K-means PCA: %.2f" %acc) # 0.53

    
    
    #clf = svm.LinearSVC()
    clf = svm.SVC(gamma='scale')
     
    clf.fit(Xpca,Ytrain)
    
    Xtestpca = pca.fit_transform(Xtest)
    ysvm = clf.predict(Xtestpca)
    
    rep = classification_report(Ytest, ysvm)
    print(rep)
    
    sys.exit(0)
    
    
    doplainkmeans = False

    #
    # Plain K-means 
    #

    if (doplainkmeans):
        problem = 'kmeans'
        filename = filename = os.path.join(models_dir, '%s.pkl' %problem)
        # Load or create model
        kmeans = loadskmodel(problem)
        if kmeans==None:
            # K-means Model
            kmeans = KMeans(n_clusters=num_classes, n_init=1, n_jobs=2, verbose=0)

            print("\nK-means Training ...")
            # Train
            try:
                kmeans.fit(Xtrain)
            except KeyboardInterrupt:
                pass

            #joblib.dump(keras, filename, compress=9)
                
        print("\n\nEvaluation ...")
        # Evaluate the K-Means clustering accuracy
        y_pred_kmeans = kmeans.predict(Xtrain)
        
        acc = evaluation(Ytrain, y_pred_kmeans)
        print("Accuracy K-means: %.2f" %acc)
    else:
        # 0.53 - 0.56
        acc = 0.53
        print("Accuracy K-means: %.2f" %acc)


    #
    # K-means with autoencoded features
    #
    
    problem = 'kmeans_ae'
    # Load or create model
    autoencoder = loadmodel(problem)
    if autoencoder==None:
        dims = [Xtrain.shape[-1], 500, 500, 2000, 10]
        init = VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform')
        autoencoder = AutoEncoder(dims, init=init)
    
    print('\nAutoEncoder')
    autoencoder.summary()
    
    print("\nAutoencoder Training ...")
    # Train
    try:
        epochs = 300
        batch_size = 256
        autoencoder.fit(Xtrain, Xtrain, batch_size=batch_size, epochs=epochs)
    except KeyboardInterrupt:
        pass

    savemodel(autoencoder, problem)

    # define encoder model
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder_3').output)

    print('\nEncoder')
    encoder.summary()

    try:

        print('\nFeature extraction')
        features_ae = encoder.predict(Xtrain)
        
        # K-means Model
        kmeans_ae = KMeans(n_clusters=num_classes, n_init=1, n_jobs=2, verbose=0)
    
        print('\nK-means AE training')
        kmeans_ae.fit(features_ae)

        print('\nK-means AE evaluation')
        y_pred_kmeans = kmeans_ae.predict(features_ae)
        acc = evaluation(Ytrain, y_pred_kmeans)
        print("Accuracy K-means AE: %.2f" %acc)
    except KeyboardInterrupt:
        pass

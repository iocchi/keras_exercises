# adapted from https://www.dlology.com/blog/how-to-do-unsupervised-clustering-with-keras/

import sys, os, datetime
import numpy as np
import argparse
import timeit

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, classification_report
from sklearn.externals import joblib
from sklearn.decomposition import PCA, IncrementalPCA
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



def savepcamodel(pca, filename):
    np.savez(filename, pcamean = pca.mean_, pcacomponents = pca.components_)
    print("PCA model saved successfully on file %s\n\n\n" %filename)

    
def loadpcamodel(pca, filename):
    try:
        filename = filename + '.npz'
        data = np.load(filename)
        pca.mean_ = data['pcamean']
        pca.components_ = data['pcacomponents']
    except IOError:
        s = "Error: cannot load data from " + filename
        print(s)
        return False

    s = "Data loaded from " + filename
    print(s)    
    return True

### main ###
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PCA example')
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
    n_components=784
    pca = PCA(n_components=n_components)
    #pca = IncrementalPCA(n_components=n_components)

    print("\nPCA with %d components ..." %n_components)
    filename = os.path.join(models_dir, 'mnist_%03d.pca' %n_components)

    if not loadpcamodel(pca,filename):
        print("\nPCA Training ...")
        t0 =  timeit.default_timer()
        pca.fit(Xtrain)
        t1 =  timeit.default_timer()
        print('   time: %.2f s' %(t1-t0))
        savepcamodel(pca, filename)
    
    print('\nPCA Tranform ...')
    t0 =  timeit.default_timer()
    Xpca = pca.transform(Xtrain)
    t1 =  timeit.default_timer()
    print('   time: %.2f s' %(t1-t0))
    
    if False:
        # equivalent to  pca.transform(Xtrain)
        from sklearn.utils.extmath import fast_dot
        td = Xtrain - means
        Xpca = fast_dot(td, components.T)

    if False:
        kmeans = KMeans(init='k-means++', n_clusters=num_classes, n_init=10, n_jobs=2, verbose=0)

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
        print("Accuracy K-means PCA: %.2f" %acc) 
        # k-means default params 
        #   100 components 0.59, 300 comp 0.51, 50 comp 0.54
        # init='k-means++', n_init=10
        #   50 components 0.51, 10 comp. 0.50

    if True:
    
        print('\nSVM training ...')
        
        #clf = svm.LinearSVC()
        clf = svm.SVC(gamma='scale')
        
        t0 =  timeit.default_timer()
        clf.fit(Xpca,Ytrain)
        t1 =  timeit.default_timer()
        print('   time: %.2f s' %(t1-t0))
            
        Xtestpca = pca.transform(Xtest)
        ysvm = clf.predict(Xtestpca)
       
        print("\n\nEvaluation ...")
        rep = classification_report(Ytest, ysvm)
        print(rep)  
        # RBF 10 comp. f1 0.94, 50 components: f1 0.98, 100 comp. 0.98 74.77 s
        # Linear 10 comp. f1 0.77, 50 components: f1  0.90  , 100 comp. 0.91
        # RBF 784 comp  0.97 448.12 s
        # RBF no PCA  0.96 457.85 s
        
    sys.exit(0)
    
    




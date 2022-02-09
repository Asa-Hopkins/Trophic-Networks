import numpy as np
import pickle
import os
from scipy.special import entr

def entropy(data): #For discrete distributions
    l = data.shape[1]
    temp = [np.unique(i, return_counts=True)[1] for i in data]
    return np.array([np.sum(entr(a/l)) for a in temp])

def Centropy(data): #For a sample from a continuous distribution
    l = data.shape[1]
    bins = int(l**0.5 + 0.5)
    hist = lambda data,bins: np.histogram(data,bins=bins)[0]
    temp = np.apply_along_axis(hist,1,data,bins)
    return np.sum(entr(temp/l),axis=1)

def process_entropy(train,test, filepath = False, discrete=True, threshold = 0.001, num = False):
    #Measures entropy of each input node in the training set, and removes the least informative
    if filepath!=False and os.path.isfile(filepath):
        e = pickle.load(open(filepath,"rb"))
    else:
        if discrete:
            e = entropy(train[0].T)
        else:
            e = Centropy(train[0].T)
        if filepath!=False:
            pickle.dump(e,open(filepath,"wb"))
    s = np.cumsum(np.sort(e))
    if num!=False:
        ind = train[2] - num
    else:
        ind = np.searchsorted(s, s[-1]*threshold)
    train[0] = np.delete(train[0], np.where(e < s[ind] - s[ind-1]), axis=1)
    test[0] = np.delete(test[0],np.where(e < s[ind] - s[ind-1]),axis = 1);
    train[2] = train[0].shape[1]; test[2] = test[0].shape[1]

def process_pca(train, test, threshold = 0.01, num = False):
    #Applies Principle Component Analysis to the training set, applying the same transform to the test set
    #Requires data to be mean centred
    cov = np.cov(train[0].T)
    eigs, vec = np.linalg.eigh(cov)
    s = np.cumsum(eigs)
    if num!=False:
        ind = train[2] - num
    else:
        ind = np.searchsorted(s,s[-1]*threshold)
    vec = vec[ind:]
    test[0] = np.dot(test[0],vec.T)
    train[0] = np.dot(train[0],vec.T)
    test[2] = test[0].shape[1]
    train[2] = train[0].shape[1]
    
def load_mnist():
    """Loads the classic MNIST data set, returns a pair of lists (one for the training set, one for the testing set) containing
    the images,labels, inputs per image, number of classes and number of images."""
    f=open("./MNIST/train-images.idx3-ubyte",'rb');images = np.frombuffer(f.read()[16:],dtype=np.uint8).reshape(60000,784)/255.0;f.close()
    f=open("./MNIST/train-labels.idx1-ubyte",'rb');labels = np.frombuffer(f.read()[8:],dtype=np.uint8);f.close()
    f=open("./MNIST/t10k-images.idx3-ubyte",'rb');test_images = np.frombuffer(f.read()[16:],dtype=np.uint8).reshape(10000,784)/255.0;f.close()
    f=open("./MNIST/t10k-labels.idx1-ubyte",'rb');test_labels = np.frombuffer(f.read()[8:],dtype=np.uint8);f.close()
    images -= np.mean(images); test_images -= np.mean(test_images)
    return [images, labels, 784, 10, 60000], [test_images, test_labels, 784, 10, 10000]

def load_balanced_emnist():
    """Loads the extended MNIST data set, returns a pair of lists (one for the training set, one for the testing set) containing
        the images,labels, inputs per image, number of classes and number of images."""
    f=open("./EMNIST/Balanced/emnist-balanced-train-images-idx3-ubyte", 'rb'); images = np.frombuffer(f.read()[16:], dtype=np.uint8).reshape(112800,784)/255.0; f.close()
    f=open("./EMNIST/Balanced/emnist-balanced-train-labels-idx1-ubyte", 'rb'); labels = np.frombuffer(f.read()[8:], dtype=np.uint8); f.close()
    f=open("./EMNIST/Balanced/emnist-balanced-test-images-idx3-ubyte", 'rb'); test_images = np.frombuffer(f.read()[16:], dtype=np.uint8).reshape(18800,784)/255.0; f.close()
    f=open("./EMNIST/Balanced/emnist-balanced-test-labels-idx1-ubyte", 'rb'); test_labels = np.frombuffer(f.read()[8:], dtype=np.uint8); f.close()
    images -= np.mean(images); test_images -= np.mean(test_images)
    return [images, labels, 784, 47, 112800], [test_images, test_labels, 784, 47, 18800]



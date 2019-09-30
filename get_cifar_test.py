#------------------------------------Import numpy and pickle module for reading cifar10 dataset-------------
import numpy as np
from six.moves import cPickle as pickle

def get_cifar_2_test_classes(class1,class2,data_path = './cifar-10-batches-py/'):#The function the test data of two classes given by args
    b1f = open(data_path+'test_batch','rb')                                      #open the test_batch file for reading test data
    b1t= pickle.load(b1f, encoding='bytes')                                      #Load data from test batch file
    b1f.close()                                                                  #Close the test batch file


    b1i= b1t[b'data']                   #Get images data from loaded data
    b1l = b1t[b'labels']                #Get labels of images from loaded data

    batch = np.array(b1i)                      #Convert data to numpy array
    labels = np.array(b1l)                     #Convert labels of data to numpy array


    c1i = batch[labels==class1]                #Read images data of two classes 
    c2i = batch[labels==class2]

    c1l = labels[labels==class1]*0             #Read labels of two classes and make them 0 and 1
    c2l = (labels[labels==class2]*0)+1

    images = np.append(c1i,c2i,axis=0)         #Append all the data to a single matrix
    labels = np.append(c1l,c2l,axis=0)         #Append all the labels to a single vector
    
    data = np.append(images,labels.reshape(len(labels),1),axis=1) #Append labels to images data
    
    np.random.shuffle(data)                   #Randomly shuffle data
    
    return data[:,0:3072],data[:,3072].astype(np.uint8) #Return images as vectors and their class labels

def get_cifar_5_test_classes(data_path = './cifar-10-batches-py/'):  #The function the test data of 5 classes given by args
    b1f = open(data_path+'test_batch','rb')                          #open the test_batch file for reading test data 
    b1t= pickle.load(b1f, encoding='bytes')                          #Load data from test batch file
    b1f.close()                                                      #Close the test batch file


    b1i= b1t[b'data']             #Get images data from loaded data
    b1l = b1t[b'labels']          #Get labels of images from loaded data

    batch = np.array(b1i)         #Convert data to numpy array
    labels = np.array(b1l)        #Convert labels of data to numpy array

    c1i = batch[labels==0]        #Get images of first 5 classes only
    c2i = batch[labels==1]
    c3i = batch[labels==2]
    c4i = batch[labels==3]
    c5i = batch[labels==4]
    

    c1l = labels[labels==0]          #Get labels of first 5 classes
    c2l = labels[labels==1]
    c3l = labels[labels==2]
    c4l = labels[labels==3]
    c5l = labels[labels==4]

    images = np.append(c1i,c2i,axis=0)     #Append all the images to a single matrix
    images = np.append(images,c3i,axis=0)
    images = np.append(images,c4i,axis=0)
    images = np.append(images,c5i,axis=0)
    
    
    labels = np.append(c1l,c2l,axis=0)       #Append all the labels to a single vector
    labels = np.append(labels,c3l,axis=0)
    labels = np.append(labels,c4l,axis=0)
    labels = np.append(labels,c5l,axis=0)
    
    data = np.append(images,labels.reshape(len(labels),1),axis=1) #Append labels to images data
    
    np.random.shuffle(data)                             #Randomly shuffle data
    
    return data[:,0:3072],data[:,3072].astype(np.uint8) #Return images as vectors and their class labels
 

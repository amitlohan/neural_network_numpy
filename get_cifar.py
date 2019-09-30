#---------------------Import numpy and pickle module for reading cifar10 dataset---------------------------
import numpy as np
from six.moves import cPickle as pickle

def get_cifar_2_classes(class1,class2,data_path = './cifar-10-batches-py/'):  #The function return all the data of two classes given by args
    b1f = open(data_path+'data_batch_1','rb')                                 #open data batches files of cifar 10
    b2f = open(data_path+'data_batch_2','rb')
    b3f = open(data_path+'data_batch_3','rb')
    b4f = open(data_path+'data_batch_4','rb')
    b5f = open(data_path+'data_batch_5','rb')

    b1t= pickle.load(b1f, encoding='bytes')     #load data from files with pickle function                                 
    b2t= pickle.load(b2f, encoding='bytes')
    b3t= pickle.load(b3f, encoding='bytes')
    b4t= pickle.load(b4f, encoding='bytes')
    b5t= pickle.load(b5f, encoding='bytes')

    b1f.close()        #close the opened data batch files
    b2f.close()
    b3f.close()
    b4f.close()
    b5f.close()

    b1i= b1t[b'data']    #Retrive images as vectors from loaded data batches
    b2i= b2t[b'data']
    b3i= b3t[b'data']
    b4i= b4t[b'data']
    b5i= b5t[b'data']

    b1l = b1t[b'labels'] #Retrieve labels from loaded data batches
    b2l = b2t[b'labels']
    b3l = b3t[b'labels']
    b4l = b4t[b'labels']
    b5l = b5t[b'labels']

    b1 = np.array(b1i)    #Convert to numpy array
    b2 = np.array(b2i)
    b3 = np.array(b3i)
    b4 = np.array(b4i)
    b5 = np.array(b5i)

    batch = np.append(b1,b2,axis=0)    #Append all the data to a single matrix
    batch = np.append(batch,b3,axis=0)
    batch = np.append(batch,b4,axis=0)
    batch = np.append(batch,b5,axis=0)

    labels = np.append(b1l,b2l,axis=0) #Append all the labels to a single matrix
    labels = np.append(labels,b3l,axis=0)
    labels = np.append(labels,b4l,axis=0)
    labels = np.append(labels,b5l,axis=0)

    c1i = batch[labels==class1]      #Extract data of only class 1 and class2
    c2i = batch[labels==class2]

    c1l = labels[labels==class1]*0    #Extract labels of only class 1 and class2 2 and make them 0 and 1
    c2l = (labels[labels==class2]*0)+1




    images = np.append(c1i,c2i,axis=0)           #append all the data to a single matrix
    labels = np.append(c1l,c2l,axis=0)           #append all the data to a single vector of labels
    
    data = np.append(images,labels.reshape(len(labels),1),axis=1) #Append data and labels
    
    np.random.shuffle(data) #randomly shuffle data
    
    return data[:,0:3072],data[:,3072].astype(np.uint8) #Return teh 2 class data and its labels

def get_cifar_5_classes(data_path = './cifar-10-batches-py/'): #The function return all the data of first 5 classes
    b1f = open(data_path+'data_batch_1','rb')                  #open data batches files of cifar 10
    b2f = open(data_path+'data_batch_2','rb')
    b3f = open(data_path+'data_batch_3','rb')
    b4f = open(data_path+'data_batch_4','rb')
    b5f = open(data_path+'data_batch_5','rb')

    b1t= pickle.load(b1f, encoding='bytes')     #load data from files with pickle function 
    b2t= pickle.load(b2f, encoding='bytes')
    b3t= pickle.load(b3f, encoding='bytes')
    b4t= pickle.load(b4f, encoding='bytes')
    b5t= pickle.load(b5f, encoding='bytes')

    b1f.close()                      #close the opened data batch files
    b2f.close()
    b3f.close()
    b4f.close()
    b5f.close()

    b1i= b1t[b'data']               #Retrive images as vectors from loaded data batches
    b2i= b2t[b'data']
    b3i= b3t[b'data']
    b4i= b4t[b'data']
    b5i= b5t[b'data']

    b1l = b1t[b'labels']            #Retrieve labels from loaded data batches
    b2l = b2t[b'labels']
    b3l = b3t[b'labels']
    b4l = b4t[b'labels']
    b5l = b5t[b'labels']

    b1 = np.array(b1i)              #Convert to numpy array
    b2 = np.array(b2i)
    b3 = np.array(b3i)
    b4 = np.array(b4i)
    b5 = np.array(b5i)

    batch = np.append(b1,b2,axis=0)        #Append all the data to a single matrix
    batch = np.append(batch,b3,axis=0)
    batch = np.append(batch,b4,axis=0)
    batch = np.append(batch,b5,axis=0)

    labels = np.append(b1l,b2l,axis=0)     #Append all the labels to a single matrix
    labels = np.append(labels,b3l,axis=0)
    labels = np.append(labels,b4l,axis=0)
    labels = np.append(labels,b5l,axis=0)

    c1i = batch[labels==0]                 #Extract data of first 5 classes
    c2i = batch[labels==1]
    c3i = batch[labels==2]
    c4i = batch[labels==3]
    c5i = batch[labels==4]
    

    c1l = labels[labels==0]                #Extract labels of first 5 classes
    c2l = labels[labels==1]
    c3l = labels[labels==2]
    c4l = labels[labels==3]
    c5l = labels[labels==4]

    images = np.append(c1i,c2i,axis=0)       #append all the data to a single matrix
    images = np.append(images,c3i,axis=0)
    images = np.append(images,c4i,axis=0)
    images = np.append(images,c5i,axis=0)
    
    
    labels = np.append(c1l,c2l,axis=0)        #append all the data to a single vector of labels
    labels = np.append(labels,c3l,axis=0)
    labels = np.append(labels,c4l,axis=0)
    labels = np.append(labels,c5l,axis=0)
    
    data = np.append(images,labels.reshape(len(labels),1),axis=1) #append labels and data
    
    np.random.shuffle(data)                     #randomly shuffle data
    
    return data[:,0:3072],data[:,3072].astype(np.uint8) #return data and its labels


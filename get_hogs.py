import numpy as np

def get_cifar_2_class_hogs(): #Load generated hogs from csv and return with labels
    data = np.loadtxt('hogs_2_classes.csv',delimiter=',')
    return data[:,0:data.shape[1]-1],data[:,data.shape[1]-1].astype(np.uint8)

def get_cifar_5_class_hogs(): #Load generated hogs from csv and return with labels
    data = np.loadtxt('hogs_5_classes.csv',delimiter=',')
    return data[:,0:data.shape[1]-1],data[:,data.shape[1]-1].astype(np.uint8)

def get_cifar_2_test_class_hogs(): #Load generated hogs from csv and return with labels
    data = np.loadtxt('hogs_2_test_classes.csv',delimiter=',')
    return data[:,0:data.shape[1]-1],data[:,data.shape[1]-1].astype(np.uint8)

def get_cifar_5_test_class_hogs(): #Load generated hogs from csv and return with labels
    data = np.loadtxt('hogs_5_test_classes.csv',delimiter=',')
    return data[:,0:data.shape[1]-1],data[:,data.shape[1]-1].astype(np.uint8)



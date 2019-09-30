#-------------------Import numpy and other custom modules for importing CIFAR10 Dataset and generating hog features--

import numpy as np
from get_cifar import *            #Module for importing CIFAR10 Dataset classes from the batch files 
from get_cifar_test import *       #Module for importing CIFAR10 Dataset classes form the test batch file
from skimage.feature import hog    #skimage.feature module for generating hog features
import sys                         #sys for fetures for clearing screen while printing information on screen

orients = 9                        #No. of orientations for generating hos features
cell_size = 8                      #Cell size for dividing image in cells

#--------------------------------------------------------------------2_classes-----------------------------------------------------------

X,Y = get_cifar_2_classes(class1=6,class2=9)   #Read 2 classes data from CIFAR10 data files
hogs = []                                      #Empty list for containing hog features
len1 = X.shape[0]                              #Number of samples in data
iteration = 0                                  #Number of sampels counter

for image in X:                                #Loop throught data for generating hogs for each image
    image = image.reshape(32,32,3)             #Reshape vector to a image
    fd = hog(image, orientations=orients, pixels_per_cell=(cell_size,cell_size),
                    cells_per_block=(1, 1), visualize=False, multichannel=True)     #Compute hogs for the image
    
    iteration+=1                              #Increment iterations counter
    if iteration%1000==0:                     #Print progress after every 1000 iterations   
        sys.stdout.write("\033[F")            #Move cursor to the previously printed line
        sys.stdout.write("\033[K")            #Clear the previously printed line
        print('Generating HoGs for 2 classes, ',(iteration*100)/len1,'%complete') #Print the progress of hogs generation
    
    hogs.append(fd)                           #Append the generated hog features to list
print('Saving to memory....')                 #Print progress
hogs = np.array(hogs)                         #convert hogs to a numpy array
print(hogs.shape)                             #Print shape of computed hogs
hogs = np.append(hogs,Y.reshape(len(Y),1),axis=1) #Append labels to hog features           
np.savetxt('hogs_2_classes.csv',hogs,delimiter=',') #Save hog features to a csv file
print('Finished.')                                  #Print finished
print('')  #Print newline for next execution
print('')  #Print newline for next execution


#--------------------------------------------------------------------2_test_classes-----------------------------------------------------------

X,Y = get_cifar_2_test_classes(class1=6,class2=9)           #Read 2 test classes data from CIFAR10 data files

hogs = []                                      #Empty list for containing hog features
len1 = X.shape[0]                              #Number of samples in data
iteration = 0                                  #Number of sampels counter

for image in X:                                #Loop throught data for generating hogs for each image
    image = image.reshape(32,32,3)             #Reshape vector to a image
    fd = hog(image, orientations=orients, pixels_per_cell=(cell_size,cell_size),
                    cells_per_block=(1, 1), visualize=False, multichannel=True)     #Compute hogs for the image
    
    iteration+=1                              #Increment iterations counter
    if iteration%1000==0:                     #Print progress after every 1000 iterations   
        sys.stdout.write("\033[F")            #Move cursor to the previously printed line
        sys.stdout.write("\033[K")            #Clear the previously printed line
        print('Generating HoGs for 2 test classes, ',(iteration*100)/len1,'%complete') #Print the progress of hogs generation   
    hogs.append(fd)                                      #Append the generated hog features to list

print('Saving to memory....')
hogs = np.array(hogs)                                    #convert hogs to a numpy array
print(hogs.shape)                                        #Print shape of computed hogs
hogs = np.append(hogs,Y.reshape(len(Y),1),axis=1)        #Append labels to hog features           
np.savetxt('hogs_2_test_classes.csv',hogs,delimiter=',') #Save hog features to a csv file


print('Finished.') #Print finished
print('')          #Print newline for next execution
print('')          #Print newline for next execution

#--------------------------------------------------------------------5_classes-----------------------------------------------------------

X,Y = get_cifar_5_classes()                        #Read 5 classes data from CIFAR10 data files
hogs = []                                      #Empty list for containing hog features
len1 = X.shape[0]                              #Number of samples in data
iteration = 0                                  #Number of sampels counter

for image in X:                                #Loop throught data for generating hogs for each image
    image = image.reshape(32,32,3)             #Reshape vector to a image
    fd = hog(image, orientations=orients, pixels_per_cell=(cell_size,cell_size),
                    cells_per_block=(1, 1), visualize=False, multichannel=True)     #Compute hogs for the image   
    iteration+=1                              #Increment iterations counter
    if iteration%1000==0:                     #Print progress after every 1000 iterations   
        sys.stdout.write("\033[F")            #Move cursor to the previously printed line
        sys.stdout.write("\033[K")            #Clear the previously printed line
        print('Generating HoGs for 5 classes, ',(iteration*100)/len1,'%complete') #Print the progress of hogs generation  
    hogs.append(fd)                                 #Append the generated hog features to list
print('Saving to memory....')                       #Print progress
hogs = np.array(hogs)                               #convert hogs to a numpy array
print(hogs.shape)                                   #Print shape of computed hogs
hogs = np.append(hogs,Y.reshape(len(Y),1),axis=1)   #Append labels to hog features           
np.savetxt('hogs_5_classes.csv',hogs,delimiter=',') #Save hog features to a csv file
print('Finished.') #Print finished
print('')          #Print newline for next execution
print('')          #Print newline for next execution


#--------------------------------------------------------------------5_classes-----------------------------------------------------------

X,Y = get_cifar_5_test_classes()                #Read 5 test classes data from CIFAR10 data files
hogs = []                                      #Empty list for containing hog features
len1 = X.shape[0]                              #Number of samples in data
iteration = 0                                  #Number of sampels counter

for image in X:                                #Loop throught data for generating hogs for each image
    image = image.reshape(32,32,3)             #Reshape vector to a image
    fd = hog(image, orientations=orients, pixels_per_cell=(cell_size,cell_size),
                    cells_per_block=(1, 1), visualize=False, multichannel=True)     #Compute hogs for the image
    
    iteration+=1                              #Increment iterations counter
    if iteration%1000==0:                     #Print progress after every 1000 iterations   
        sys.stdout.write("\033[F")            #Move cursor to the previously printed line
        sys.stdout.write("\033[K")            #Clear the previously printed line
        print('Generating HoGs for 5 test classes, ',(iteration*100)/len1,'%complete') #Print the progress of hogs generation
    hogs.append(fd)                                     #Append the generated hog features to list
print('Saving to memory....')                           #Print progress
hogs = np.array(hogs)                                   #convert hogs to a numpy array
print(hogs.shape)                                       #Print shape of computed hogs
hogs = np.append(hogs,Y.reshape(len(Y),1),axis=1)       #Append labels to hog features           
np.savetxt('hogs_5_test_classes.csv',hogs,delimiter=',')#Save hog features to a csv file
print('Finished.') #Print finished


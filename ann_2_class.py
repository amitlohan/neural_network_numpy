#--------------------------import numpy and custom libraries--------------------------------
import numpy as np
import matplotlib.pyplot as plt 
from get_hogs import * #import the functions to retrieve hog features saved in csv files
from data_loader import * #import data_loader class for retirieving batches of data
from nn_model import nn_model


data_X,labels_Y = get_cifar_2_class_hogs() #Retrive hog features and labels for the training images 
X_test,Y_test = get_cifar_2_test_class_hogs() #Retirves hos featrues and labels for the test images

#--------------------Print data size----------------------------------------
print('No. of traing samples:',data_X.shape[0])
print('No. of test samples:',X_test.shape[0])


#-----------------Create a model of 3 layers using the class nn_model ------------------
model = nn_model(np.array([data_X.shape[1],360,190,5]),0.0001) #Create a model of 3 layers
model.print_info()


epochs = 100  #Number of epochs
lr = 0.2     #Learning rate
model.update_lr(lr) #Update learning rate of the model
batch_size=10  #Batch size for training

iteration_num = 0 #Keeps a count of the number of batches processed 
batch_loss = []   #Empty container for storing batch losses
epoch_loss = []   #Empty container for storing epoch losses

for i in range(epochs):  #Main loop for training
    batch_no = 0
    for X,Y in data_batches(data_X,labels_Y,batch_size): #Training loop for processing batches
        outs = model.forward(X.T)                        #Get output of model's forward pass
        loss,grads = model.loss(outs,Y.T)                #Compute loss and its gradient
        iteration_num+=1;                                #Increment the batch counter
        batch_loss.append(loss)                          #Store each batch loss in a list
        backw = model.backward(grads)                    #Backward pass with computed gradients

    outs = model.forward(data_X.T)      #Forward pass the whole data after each epoch
    loss,grads = model.loss(outs,labels_Y) #Compute loss and gradient for the training data
    epoch_loss.append(loss)                #Store epoch loss
   

    ypred = np.argmax(outs,0)     #Get predicted labels
    ypred_test = np.argmax(model.forward(X_test.T),0) #Get predicted labels for test data
    acc = (np.sum(ypred==labels_Y)*100)/len(labels_Y) #Calculate train accuracy
    acc_val = (np.sum(ypred_test==Y_test)*100)/len(Y_test) #Calculate test accuracy
    print("epoch:",i," loss:",loss,'Train accuracy:',acc,'%',' Validation Accuracy:',acc_val,'%')

    
#Plot batch loss
fig = plt.figure(figsize=(30,30))                     
fig.add_subplot(1,2,1)
plt.title('Batch Loss',fontsize=30)
plt.plot(np.linspace(1,iteration_num,iteration_num),np.array(batch_loss))                                    

#Plot epoch loss
fig.add_subplot(1,2,2)
plt.title('Epoch Loss',fontsize=30)
plt.plot(np.linspace(1,epochs,epochs),np.array(epoch_loss))  

acc = np.sum(ypred==labels_Y)/len(labels_Y) #Compute train accuracy
print("Final Train Accuracy:",acc*100,'%')    #Print train accuracy

ypred_test = np.argmax(model.forward(X_test.T),0) #Get predicted labels for test data
acc_val = (np.sum(ypred_test==Y_test)*100)/len(Y_test) #Calculate test accuracy
print("Final Test Accuracy:",acc_val,'%')    #Print train accuracy








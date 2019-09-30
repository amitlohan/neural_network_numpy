import numpy as np

#--------------------Yields batches of data from X and labels from Y, batch size give by batch_size------------
def data_batches(X,Y,batch_size):  
 
   data_len = len(Y)  #Length of data
   low = 0           #variable to track the first element of a data batch
   high = batch_size #variable to track the last element of a data batch
    
   while low<data_len:   #Loop throught the whole dataset
      yield X[low:min(high,data_len)],Y[low:min(high,data_len)].astype(np.uint8) #yield batches from the data matrix
      low+=batch_size       #Increment the index for first element of next batch
      high+=batch_size      #Increment the index for last element of next batch




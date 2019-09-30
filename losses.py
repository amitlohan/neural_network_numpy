import numpy as np


#-------------Cross entropy loss-------------------------------------------
def cross_entropy(ytrue,ypred):
    y_capped = np.maximum(0.00000000001,np.minimum(ypred,0.99999999999999)) #Restrict the y pred from becoming 0 or 1 
    yl1 = -np.log(y_capped)
    y_hot = (np.eye(yl1.shape[0])[ytrue.astype(np.uint8)]).T      #One hot encode labels
    return sum(sum(np.multiply(yl1,y_hot))/len(ytrue))

#------------Gradient of cross entropy loss--------------------------------   
def cross_entropy_grad(ytrue,ypred):
    y_hot = (np.eye(ypred.shape[0])[ytrue]).T #one hot encode labels
    grad =  ypred-y_hot
    return grad                      #Return cross entropy gradients
    







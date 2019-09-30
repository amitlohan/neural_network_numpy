#----------------------------------------Import numpy------------------------
import numpy as np


#--------------------------Relu activation fucntion for use in nn layers-----
class relu():
    def __init__(self):
        pass

    def forward(self,In_batch):  #Take a batch from the outputs of layer
        activation_values = np.maximum(0,In_batch) #Compute activation values according to the ReLU activation
        activation_caches = np.zeros_like(In_batch) #Initialize varialbe for storing caches for backward pass
        activation_caches[activation_values>0] = 1 #compute caches values for backward pass gradients
        return activation_values,activation_caches #Return the activation values and caches for backward pass
    
    def backward(self,grads_in,caches):            #Backward pass of ReLU activation function
        return np.multiply(caches,grads_in)        #Return the calculates gradients w.r.t layer outputs


#-----------------------------Softmax activation function of use in last layer of nn---------
class softmax():
    def __init__(self):
        pass

    def forward(self,In_batch):      #Forward pass of softmax activation functions
        In_batch = In_batch - np.amax(In_batch,axis=0) #Subtract maximum value for stable softmax
        exps = np.exp(In_batch)         #Compute exponentials of scores give by the layer
        exp_sums = np.sum(exps,axis=0)  #Sum of exponentials for softmax denominator
        activation_values = exps/exp_sums #Activation values or probabilities
        activation_caches = activation_values #Activation aches to be used for backward pass of softmax
        return activation_values,activation_caches #Return activation values and caches
    
    def backward(self,grads_in,probs): #Backward pass of softmax activation function
        n=np.size(probs,0)             #size of stored caces or number of outputs of the softmax 
        grad_probs=np.multiply(grads_in,probs) #Product of grads and probabilites give intermediate
        cprobs = 1 - probs                     #matrix for gradient computation
        grads = np.multiply(grad_probs,cprobs) #Intermediate matrix for gradients computation
        for i in range(n-1):                   #Loops to calculate gradient w.r.t softmax inputs 
            grads=grads-np.multiply(probs,np.roll(grad_probs,i+1,axis=0))
            #Above line implements e.g. p1(1-p1)grad1 -p1p2*grad2 - p1p3*grad3 etc.   
        return grads #return computed gradients w.r.t outputs of the layer
    
class pass_through():   #A dummy activation function which only acts as a placeholder 
    def __init__(self): #when activation is not required for a layer e.g. in last layer when loss is external
        pass

    def forward(self,In_batch):  #Dummy forward, acts as palceholder in loop through layers
        return In_batch,In_batch #returns same as received parameters
    
    def backward(self,grads_in,probs): #Dummy backward, acts as placeholder 
        return grads_in  

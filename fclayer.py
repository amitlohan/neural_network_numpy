#----------------------------------------Import numpy and activation functions from custom module------------------------------------
import numpy as np
from activations import *

class fclayer:
    def __init__(self,No,Ni,train_rate,activation,name): # No is number of neurons in the layer, Ni is the nuber of inputs to each neuron.
        self.No = No                                     #Initialized variable for number of outputs of the fully connected layer
        self.Ni = Ni
        self.train_rate=train_rate                       #Training rate for the layer
        self.name = name                                 #Layer name
        
        self.weights = np.random.rand(No,Ni)*0.01        #Initialize weights of the layer
        self.b = np.random.rand(No)*0.01                 #Initialize biases of the layer
   
        self.activation = activation                     #Activation function for the layer as received in an argument
        
        self.x_cache = np.zeros(Ni)                      #Cache for calculating upstream gradients
        self.act_cache = np.zeros(Ni)                    #Caches for calculating gradient w.r.t activation function inputs
        
        self.dW = np.zeros([No,Ni])  #Create an array to store gradient of loss w.r.t. W 
        self.db = np.zeros(No)     #Create an array to store gradients of loss w.r.t b
        
            
    def forward(self,In_batch):             #Forward function of the layer
        if(np.shape(In_batch)[0]!=self.Ni):   #Checks whether the input received is of correct dimensions for the layer
            print("\nError in forward: Input dimensions mismatch in "+self.name+"\n") #Prints error message if the input dimensions mismatch
            print("Expected:",self.Ni)        #Expected input size
            print("Received:",np.shape(In_batch)[0]) #Received input size
            return
        else:
            out1 = np.add((np.matmul(self.weights,In_batch)),(self.b).reshape(self.No,1)) #Calculate layer output using w's and biases
            self.x_cache = In_batch                               #Store caches for backward pass
            (out2,self.act_cache) = self.activation.forward(out1) #Get output of activation function and caces for backward pass
            return out2                                           #Return the output activations
            
    
    def backward(self,grad_in):               #Backward function for the layer
        if(np.shape(grad_in)[0]!=self.No):    #Compare shape of the input gradients with expected shape of the gradients
            print("\nError in forward: Input dimensions mismatch in "+self.name+"\n") #Error message if the shapes mismatch
            print("Expected:",self.No)      #Expected inputs size
            print("Received:",np.shape(grad_in)[0]) #Size of received gradients
            return
        else:
            grad_act = self.activation.backward(grad_in,self.act_cache) #Compute gradients of inputs to activation w.r.t layer outputs
            self.db = np.average(grad_act,axis=1)            #Compute average gradients for updating biases
            self.dW = (np.matmul(self.x_cache,grad_act.T)/grad_in.shape[1]).T #Compute gradients for updating weights
            
            back =(np.matmul(grad_act.T,self.weights)).T #Compute upstream gradients
            
            self.weights = self.weights-self.train_rate*self.dW #Update weights of the current layer
            self.b = self.b-self.train_rate*self.db             #Update biases
            return back                                         #Return upstream gradients

#--------------------------import numpy and custom libraries--------------------------------
import numpy as np
from fclayer import fclayer #fclayer contains the class for a fully connected layer
from activations import * #import activation functions softmax, relu and dummy
from losses import * #import the cross_entropy loss and cross_entropy_gradient funcitons


#------------nn_model class creates the structure of  network by adding layers and---------
#------------activations as specified by the structure parameter passed to --init__()-----
class nn_model(): 
    def __init__(self,structure,l_rate): 
        
        self.n_layers= len(structure)-1 #Number of layers in the network
        self.layers = []                #empty container for storing layer objects
        self.sq_fn = softmax()          #Sqashing function to be used in the last layer
    
        for i in range(self.n_layers-1):  #Loop to add layers to network expect last layer
            name="Layer{}"                #All the layers except last layer have relu activations
            name=name.format(i+1)
            self.layers.append(fclayer(structure[i+1],structure[i],l_rate,relu(),name)) #No,Ni,train_rate,activation,batch_size,name
        name="Layer{}" 
        name=name.format(i+2)            #Name every layers for the purpose of displaying
        self.layers.append(fclayer(structure[self.n_layers],structure[self.n_layers-1],l_rate,pass_through(),name))
        #Above line of code adds a last layer to the network which has no nonlinearity layer
    
    def print_info(self):  #Prints model layers and structure
            for i in range(self.n_layers): 
                print("\nLayer name:"+(self.layers[i]).name)
                temp = "Inputs:{}\nOutpus:{}"
                temp=temp.format(self.layers[i].Ni,self.layers[i].No)
                print(temp)
                print(self.layers[i].activation)
    
    def forward(self,in_batch):  #Forward pass through the network
        for i in range(self.n_layers):
            in_batch=(self.layers[i]).forward(in_batch)
        return in_batch
    
    def backward(self,in_grads):   #Backward pass throught the network
        for i in range(self.n_layers):
            in_grads=(self.layers[self.n_layers-1-i]).backward(in_grads)
        return in_grads
    
    def plot_weights(self):     #Plots the mean of weights of each layer of the model
        layers_i = np.arange(self.n_layers)
        weights = []
        for i in layers_i:
            weights.append(np.mean(np.abs(self.layers[i].weights)))        
        weights = np.array(weights)
        plt.plot(layers_i,weights)
        plt.show()
        
    def update_lr(self,lr):  #Updates the learning rate for each layer
        for i in range(self.n_layers):
            self.layers[i].train_rate = lr
    
    def loss(self,outputs,labels):#Cross entropy loss and its derivative
        probs,_ = self.sq_fn.forward(outputs)
        loss = cross_entropy(labels,probs)
        grads = cross_entropy_grad(labels,probs)
        return loss,grads

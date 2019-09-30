import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from get_cifar import *                       #Custom DataLoader for CIFAR10 2 and 5 classes
from get_cifar_test import *                  #CIFAR10 test batch loader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') #Device to be used for computations

def data_batches(X,Y,batch_size):            #Takes Dataset and Labels and retuns uniform sized batches given by batch_size
    data_len = X.shape[0]                    #Number of samples in data
    low = 0                                  #Variable to be used for reading batch from dataset
    high = batch_size                       
    
    while low<data_len :                     #This loop yields batches from dataset
        yield X[low:min(high,data_len)],Y[low:min(high,data_len)].type(torch.long)
        low+=batch_size
        high+=batch_size

X_train,Y_train = get_cifar_5_classes()      #Reads data from CIFAR10 Batches for 5 Classes
X_test,Y_test = get_cifar_5_test_classes()   #Reads test data from CIFAR10 test Batch

X_train = X_train.reshape(X_train.shape[0],3,32,32)  #Reshape the matrix of flattened images in to 3x32x32 images
X_test = X_test.reshape(X_test.shape[0],3,32,32)     #Reshape the test data into tensor of rgb images
X_train= torch.Tensor(X_train)                       #Convert data to torch Tensor
Y_train= torch.Tensor(Y_train)                       #Convert Labels data to torch Tensor
X_test= torch.Tensor(X_test)                         #Convert test data to torch Tensor
Y_test= torch.Tensor(Y_test)                         #Convert test labels to torch Tensor

class ConvNet(nn.Module):                            #Define the class for mode to be used for classication
    def __init__(self, num_classes=5):
        super(ConvNet, self).__init__()              #Initialize the members ob
        self.layer1 = nn.Sequential(                 #Define the first layer of model 
            nn.Conv2d(3,30, kernel_size=3, stride=1, padding=1), #Convolutional layer with input channel 3 and output channels 30
            nn.BatchNorm2d(30),                      #Batch Norm Layer 
            nn.ReLU(),                               #Nonlinearity Relu
            nn.MaxPool2d(kernel_size=2, stride=2))   #Max pooling layer reducing the size of feature map to half
        self.layer2 = nn.Sequential(                 #Layer two with structure similar to the above layer
            nn.Conv2d(30, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(                 #Layer 3 with structure similar to above layers
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(4*4*64, num_classes)     #Fully connected layer with outputs equal to number of classes
        
    def forward(self, x):                            #Forward funciton for the model
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

model = ConvNet().to(device)      #Create the model object

batch_size = 10       #Batch size for training
epochs = 20           #Number of epochs
learning_rate=0.00005  #Learning rate for the model
criterion = nn.CrossEntropyLoss()  #Loss criterion for trainig
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
i = 0                 #Counter for counting the number of batches processed

for epoch in range(epochs):      #The training loop
    for images,labels in data_batches(X_train,Y_train,batch_size): #Retrives batches from dataloader
        images = images.to(device)     #Transfers the images tensor to device (CPU or GPU)
        labels = labels.to(device)     #Labels for training data
        
        # Forward pass
        outputs = model(images)        #Forward pass the data batches
        loss = criterion(outputs, labels) #Calculates loss (Cross entropy in this model)
        
        # Backward and optimize
        optimizer.zero_grad()          #Clears the gradients from previous backward passes
        loss.backward()                #backward pass the gradient of loss
        optimizer.step()               #Update parameters of the model
        
        i+=1
        if (i+1) % 2000 == 0:
            print ('Epoch [{}/{}], Step [{}], Loss: {:.4f}'.format(epoch+1,epochs, i+1,loss.item()))

model.eval()  #Enter in evaluation mode disabling the gradient computations for parameters
with torch.no_grad(): 
    correct = 0   #Initialize counter for storing the correct predictions
    total = 0     #Initialize the counter for storing total number of test instances processed
    for images, labels in data_batches(X_test,Y_test,batch_size): #Reads test data batches
        images = images.to(device)     #Transfers images tensor to device used for computation
        labels = labels.to(device)     #Transfers labels tensor to device used for computation
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)  #Get predicted labels from calculated scores at the output
        total += labels.size(0)                    #Aggregate total samples processed
        correct += (predicted == labels).sum().item() #Aggregate total correct labels from predicted outputs

    print('Test Accuracy: {} %'.format(100 * correct / total)) #Calculate test accuracy


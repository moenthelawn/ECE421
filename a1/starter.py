import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
def loadData():
    with np.load('notMNIST.npz') as data :
        Data, Target = data ['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

#============================Question 1 : Linear Regression====================
def MSE(W, b, x, y, reg):
    # Your implementation here
        
    N = len(y)
    Total = 0
    i = 0
    #You wanna sum the inside and then multiply 1/2N after the loop exits 
    for i in range(N):
        X_sliced = x[i,:,:]
        X_sliced = np.reshape(X_sliced, (1,np.product(X_sliced.shape)))
        
        Total += np.square(((np.matmul((W),np.transpose(X_sliced)) + b[0][1])) - y[i])

    Total *= (1/(2*N)) 
    Total += (reg/2) * np.matmul(W, np.transpose(W))
    return Total 

def meanSquareError(W,x,y): 
    #This function returns the minimized weights for the mean square 
    x_t = np.transpose(x)
    x_dagger = np.matmul((np.invert(np.matmul(x_t,x))),x_t)
    return np.matmul(x_dagger,W)
    
def gradMSE(W, b, x, y, reg):
    N = len(y)
    mse_gradient_weights = np.random.normal(0,1,np.shape(W)) #Matrix to hold the gradients wrt weights 
    mse_gradient_biases = np.random.normal(0,1,np.shape(W))#Declare a random set of matrix values for the biases 
    
    #You wanna sum the inside and then multiply 1/N after the loop exits 
    for i in range(N): 
        
        X_sliced = x[i,:,:]
        X_sliced = np.reshape(X_sliced, (1,np.product(X_sliced.shape)))
        mse_gradient_weights += np.dot((((np.matmul((W), np.transpose(X_sliced))) + b[0][1])-y[i]),X_sliced) 
        mse_gradient_biases += (((np.matmul((W), np.transpose(X_sliced))) + b[0][1])-y[i]) 
        
    mse_gradient_weights *= 1/N 
    mse_gradient_weights += reg * W
    
    mse_gradient_biases *= 1/N
    #mse_gradient_biases += reg * W
    
    return mse_gradient_weights,mse_gradient_biases
def getSign(number): 
    #This function will return the sign of a number 
    if number >= 0: 
        return 1 
    else: 
        return 0
    
    
def accuracy(x,b,y): 
    #This function will return the correct accuracy of a classifier 
    N = len(y)
    correctPrediction = 0 
    for i in range(N): 
        X_sliced = x[i,:,:]
        X_sliced = np.reshape(X_sliced, (1,np.product(X_sliced.shape)))
        y_hat = getSign(np.matmul(W,np.transpose(X_sliced)) + b[0][1])
        
        y_expected = y[i]   
        
        if y_hat == y_expected: 
            correctPrediction += 1
            
    accuracy = 100 * (correctPrediction / N)
    print("The accuracy of the data set is " + str(accuracy))
        
   
def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS):
    #Added the ability to loop through and 
   
    j = 0 
    for _reg in reg:
        error_history = []
        for i in range(iterations):
            mse = MSE(W,b,trainingData,trainingLabels,_reg)
            error_history.append(mse[0]) 
            if mse <= EPS: 
                break 
            mse_gradient_weights, mse_gradient_biases = gradMSE(W,b,trainingData, trainingLabels,_reg)
            
            W -= mse_gradient_weights*alpha 
            b -= mse_gradient_biases*alpha
            
           
        
        f = plt.figure(j)
        accuracy(trainingData,b,trainingLabels)
        string_plot = "Regularization Rate of " + str(_reg) + " with error " + str(mse[-1])
        plt.plot(error_history) 
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        plt.title (string_plot)
        f.show()
        plt.savefig(str(j))
        
        print(mse[-1])
        j += 1
    return W,b

#==================Question 2: Logistic Regression=============================
def crossEntropyLoss(W, b, x, y, reg):
    total = 0
    N = len(y)
    for i in range(N):  
        X_sliced = x[i,:,:]
        X_sliced = np.reshape(X_sliced, (1,np.product(X_sliced.shape)))
        sigmoid_value = sigmoid(W, X_sliced, b[0][1])
        #The next line commented out is where it breaks. Try debugigng and you'll see that the output of the 
        #sigmoid function is always 1 lmao. i'm sad
        #total += (-1 * y[i] * np.log(sigmoid_value)) - ((1 - y[i])*(np.log(1 - sigmoid_value)))
        total = total * (1/N)
    
    total += (reg/2) * np.matmul(W, np.transpose(W))
    
    print(total)
    return total 
    
#def gradCE(W, b, x, y, reg):
    
#Helper function to calculate the value of the sigmoid function 1/(1 + e ^ Wtransposex + b)^-1
# W is a matrix, x and b are single values
def sigmoid(W, x, b):
    sigmoid_output = 0 
    raised_term = np.matmul(W, np.transpose(x)) + b
    exponential_term = np.exp(-raised_term) 
    sigmoid_output = 1 / (1 + exponential_term)
    
    return sigmoid_output 

# =============================================================================
# def crossEntropyLoss(W, b, x, y, reg):
#     # Your implementation here
# 
# def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS):
    #Your Implementaiton here 
# def gradCE(W, b, x, y, reg):
#     # Your implementation here
# 
# def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS):
#     # Your implementation here
# 
# def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
#     # Your implementation here
# =============================================================================
    
trainData,validData,testData,trainTarget,validTarget,testTarget = loadData()
 
W = np.random.rand(784,1)
W = np.reshape(W, (1,np.product(W.shape)))
b = np.ones(np.shape(W))


crossEntropyLoss(W,b,trainData, trainTarget, 0)
#MSE(W,b,testData,testTarget,0.1)
#mse_gradient_weights, mse_gradient_biases = gradMSE(W,1,testData,testTarget,0.1)

#grad_descent(W, b, validData, validTarget, 0.005, 5000, {0.001,0.1,0.5}, 0.000001) 

#plt.scatter(np.matmul(np.transpose(W),testData)) + b, testTarget)
#plt.show()

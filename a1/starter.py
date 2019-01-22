import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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

def MSE(W, b, x, y, reg):
    # Your implementation here
        
    N = len(y)
    Total = 0
    i = 0
    #You wanna sum the inside and then multiply 1/2N after the loop exits 
    for i in range(N):
        X_sliced = x[i,:,:]
        X_sliced = np.reshape(X_sliced, (1,np.product(X_sliced.shape)))
        
        Total += np.square(((np.matmul((W),np.transpose(X_sliced)) + b - y[i])))

    Total *= (1/(2*N)) 
    Total += (reg/2) * np.matmul(W, np.transpose(W))
    return Total 
    
def gradMSE(W, b, x, y, reg):
    N = len(y)
    mse_gradient_weights = np.zeros(np.shape(W)) #Matrix to hold the gradients wrt weights 
    mse_gradient_biases = 0
    
    
    #You wanna sum the inside and then multiply 1/N after the loop exits 
    for i in range(N): 
        X_sliced = x[i,:,:]
        X_sliced = np.reshape(X_sliced, (1,np.product(X_sliced.shape)))
        mse_gradient_weights += np.dot((((np.matmul((W), np.transpose(X_sliced))) + b)-y[i]),X_sliced) 
        mse_gradient_biases += (((np.matmul((W), np.transpose(X_sliced))) + b)-y[i]) 
   
    mse_gradient_weights *= 1/N 
    mse_gradient_weights += reg * W
    
    mse_gradient_biases *= 1/N
    #mse_gradient_biases += reg * W
    
    return mse_gradient_weights,mse_gradient_biases
   
def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS):
    
   
    error_history = []
    
    for i in range(iterations):
        mse = MSE(W, b, trainingData, trainingLabels,reg)
        error_history.append(mse) 
        if mse <= EPS: 
            break 
        mse_gradient_weights, mse_gradient_biases = gradMSE(W,b,trainingData, trainingLabels,reg)
        W += mse_gradient_weights*alpha 
        b += mse_gradient_biases*alpha
    plt.plot(error_history)
    plt.show()
        
    return W,b
    
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
 
W = testData[0,:,:]
W = np.reshape(W, (1,np.product(W.shape)))
b = np.zeros(np.shape(W))

#MSE(W,1,testData,testTarget,0.1)
#mse_gradient_weights, mse_gradient_biases = gradMSE(W,1,testData,testTarget,0.1)

grad_descent(W, 1, trainData, trainTarget, 0.01, 5000, 0.1, 0.000001) 

#plt.scatter(np.matmul(np.transpose(W),testData)) + b, testTarget)
#plt.show()

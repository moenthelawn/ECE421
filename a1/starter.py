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
    #You wanna sum the inside and then multiply 1/2N after the loop exits 
    for i in range(N):
        X_sliced = x[i,:,:]
        X_sliced = np.reshape(X_sliced, (1,np.product(X_sliced.shape)))
        
        Total += np.square(((np.matmul((W),np.transpose(X_sliced)) + b)-y[i]))

    Total *= (1/(2*N)) 
    Total += (reg/2) * np.matmul(W, np.transpose(W))
    return Total 
    
def gradMSE(W, b, x, y, reg):
    N = len(y)
    mse_gradient = 0 
    #You wanna sum the inside and then multiply 1/N after the loop exits 
    for i in range(N): 
        X_sliced = x[i,:,:]
        X_sliced = np.reshape(X_sliced, (1,np.product(X_sliced.shape)))
        mse_gradient += (((np.matmul((W),np.transpose(X_sliced)) * X_sliced) + b)-y[i])
    
    mse_gradient *= 1/N 
    mse_gradient = reg * np.matmul(W, np.transpose(W))
    
    return mse_gradient
   # Your implementation here
    
# =============================================================================
# def crossEntropyLoss(W, b, x, y, reg):
#     # Your implementation here
# 
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


MSE(W,1,testData,testTarget,0.1)
gradMSE(W,1,testData,testTarget,0.1)


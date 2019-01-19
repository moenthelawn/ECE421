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
    N =len(y) #Length of the output training set (3500 in this case ) 
    Total = 0
    for i in range(N):
        Total += (1/(2*N)) * np.square((y[:,i] - np.transpose(W) * x[:,i] + b))
    Total += (reg/2) * np.sum(W) 
    return Total 
        
#This is a comment
# =============================================================================
# def gradMSE(W, b, x, y, reg):
#     # Your implementation here
# 
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
 
W = np.random.rand(1,len(trainData))


MSE(W,1,testData,testTarget,0.1)


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

def MSE(W, b, x, y, reg):
    # Your implementation here
        
    N = len(y)
    Total = 0
    i = 0
    #You wanna sum the inside and then multiply 1/2N after the loop exits 
    for i in range(N):
        X_sliced = x[i,:,:]
        X_sliced = np.reshape(X_sliced, (1,np.product(X_sliced.shape)))
        
        Total += np.square(((np.matmul((W),np.transpose(X_sliced)) + b)) - y[i])

    Total *= (1/(2*N)) 
    Total += (reg/2) * np.matmul(W, np.transpose(W))
    return Total[0][0]
def MSE_Normal(W, b, x, y, reg):
    # Your implementation here
        
    N = len(y)
    Total = 0
    i = 0
    #You wanna sum the inside and then multiply 1/2N after the loop exits 
    for i in range(N):
        X_sliced = x[i,:,:]
        X_sliced = np.reshape(X_sliced, (1,np.product(X_sliced.shape)))
        Total += np.square(((np.matmul((W),np.transpose(X_sliced)) + b)) - y[i])

    Total *= (1/(N)) 
    return Total 
def meanSquareError_normalWeights(W,X,y): 
    #This function returns the minimized weights for the mean square 
    x = X.reshape(np.shape(X)[0],-1)
    biases = np.ones((np.shape(y)))
    
    x = np.concatenate((biases,x),axis=1)
    x_t = np.transpose(x)
    
    x_inverted = np.linalg.inv(np.matmul(x_t,x))
    wlin = np.matmul(np.matmul(x_inverted,x_t),y)
    return wlin[:][1:],wlin[0][0]
    
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
    mse_gradient_biases += reg * W
    #mse_gradient_biases += reg * W
    
    return mse_gradient_weights,mse_gradient_biases
def getSign(number): 
    #This function will return the sign of a number 
    if number >= 0: 
        return 1 
    else: 
        return 0
    
    
def accuracy(x,W,b,y): 
    #This function will return the correct accuracy of a classifier 
    N = len(y)
    correctPrediction = 0 
    for i in range(N): 
        X_sliced = x[i,:,:]
        X_sliced = np.reshape(X_sliced, (1,np.product(X_sliced.shape)))
        y_hat = getSign(np.matmul(W,np.transpose(X_sliced)))
        y_expected = y[i]   
        
        if y_hat == y_expected: 
            correctPrediction += 1
            
    accuracy = 100 * (correctPrediction / N)
    return accuracy
        
def grad_descent_NormalEquation(W,trainingData,trainingLabels): 
       
   W_t,b = meanSquareError_normalWeights(W,trainingData,trainingLabels)       
   return np.transpose(W_t), b
    
def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS):
    #Added the ability to loop through and 
   
    j = 0 
    error_history = []
    for i in range(iterations):
        mse = MSE(W,b,trainingData,trainingLabels,reg)
        error_history.append(mse) 
        if mse <= EPS: 
            break 
        mse_gradient_weights, mse_gradient_biases = gradMSE(W,b,trainingData, trainingLabels,reg)
        
        W -= mse_gradient_weights*alpha 
        b -= mse_gradient_biases*alpha
    # mse = MSE(np.transpose(grad_descent),b,trainingData,trainingLabels,_reg)
   # f = plt.figure(j)
  #  accuracy(trainingData,W,b,trainingLabels)
   # string_plot = "Learning Rate of " + str(_reg) + " with error " + str(mse[-1])
    
    # plt.plot(error_history) 
    #plt.xlabel('Iterations')
    #plt.ylabel('Error')
   # plt.title (string_plot)
   # f.show()
    #plt.savefig(str(j))
    
    #print(mse[-1])
    #j += 1
    return W,b
def crossEntropyLoss(W, b, x, y, reg):
    total = 0
    N = len(y)
    for i in range(N):  
        X_sliced = x[i,:,:]
        X_sliced = np.reshape(X_sliced, (1,np.product(X_sliced.shape)))
        sigmoid_value = sigmoid(W, X_sliced, b)
        total += (-1 * y[i] * np.log(sigmoid_value)) - ((1 - y[i])*(np.log(1 - sigmoid_value)))
        total = total * (1/N)
    
    total += (reg/2) * np.matmul(W, np.transpose(W))
    
    print(total)
    return total 
    
def sigmoid(W, x, b):
    sigmoid_output = 0 
    raised_term = np.matmul(W, np.transpose(x)) + b
    exponential_term = np.exp(-raised_term) 
    sigmoid_output = 1 / (1 + exponential_term)
    
    return sigmoid_output 

#def crossEntropyLoss(W, b, x, y, reg):
#Your implementation
    
#def gradCE(W, b, x, y, reg):
#Your implementation

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
W_Normal, bias_Normal = grad_descent_NormalEquation(W,trainData,trainTarget)
#MSE(W,1,testData,testTarget,0.1)

#mse_gradient_weights, mse_gradient_biases = gradMSE(W,1,testData,testTarget,0.1)

#W_Gradient, b_gradient = grad_descent(W, 1, trainData, trainTarget, 0.005, 5000, 0.001, 0.000001) 

#accuracy_test = accuracy(testData,W_Gradient,b_gradient,testTarget)
#error_test = MSE(W_Gradient,b_gradient,testData,testTarget, 0.001)
#print("The accuracy of the gradient descent against the test data is ", str(accuracy_test), " with error ", str(error_test))

#accuracy_test = accuracy(validData,W_Gradient,b_gradient,validTarget)
#error_test= MSE(W_Gradient,b_gradient,validData,validTarget, 0.001)
#print("The accuracy of the gradient descent against the test data is ", str(accuracy_test), " with error ", str(error_test))
#print("------------------------------------------------------") 

#accuracy_test_normal = accuracy(testData,W_Normal,bias_Normal,testTarget)
#error_test= MSE(W_Normal,bias_Normal,testData,testTarget, 0)
#print("The accuracy of the Normal Equation against the test data is ", str(accuracy_test_normal), " with error ", str(error_test))

#accuracy_valid_normal = accuracy(validData,W_Normal,bias_Normal,validTarget)
#error_valid= MSE(W_Normal,bias_Normal,validData,validTarget, 0)
#print("The accuracy of the Normal Equation against the valid data is ", str(accuracy_valid_normal), " with error ", str(error_valid))


#plt.scatter(np.matmul(np.transpose(W),testData)) + b, testTarget)
#plt.show()

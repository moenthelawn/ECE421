import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from sklearn.utils import shuffle
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
    X_flattened = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
    y_pred = np.matmul(X_flattened,W) + b
    mse = np.sum(np.square(y_pred - y))/(2.0*x.shape[0])
    return (mse + (reg/2.0)*(np.linalg.norm(W)**2))


def gradMSE(W, b, x, y, reg):
    N = len(y)
    mse_gradient_weights = np.random.normal(0,1,np.shape(W)) #Matrix to hold the gradients wrt weights 
    mse_gradient_biases = np.random.normal(0,1,np.shape(W))#Declare a random set of matrix values for the biases 
    X_flattened = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
    y_pred = np.matmul(X_flattened,W) + b
    yinside = y_pred - y
    mse_gradient_weights = np.matmul(np.transpose(X_flattened),yinside)/x.shape[0] + (reg)*(np.linalg.norm(W))
    mse_gradient_biases = np.sum(yinside)/x.shape[0]  
    return mse_gradient_weights,mse_gradient_biases

def crossEntropyLoss(W, b, x, y, reg):
    X_flattened = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
    XW_matrix = np.matmul(X_flattened,W)
    y_pred = sigmoid(XW_matrix + b)
    cross_entropy = np.sum((np.multiply(-1.0 * y,np.log(y_pred))) - np.multiply((1.0 - y),np.log(1.0-y_pred)))/x.shape[0]
    total_loss = cross_entropy + (reg/2)*(np.linalg.norm(W)**2)
    return total_loss

def gradCE(W, b, x, y, reg): 
    grad_ce_weights = 0 
    grad_ce_biases = 0
    X_flattened = x.reshape(x.shape[0],x.shape[1] * x.shape[2])
    XW_matrix = np.matmul(X_flattened,W)
    y_pred = sigmoid(XW_matrix + b)
    grad_ce_weights = np.matmul(np.transpose(X_flattened), y_pred-y)/x.shape[0] + (reg)*(np.linalg.norm(W))
    grad_ce_biases =  np.mean((y_pred - y))
    return grad_ce_weights, grad_ce_biases  
    
#Helper function to evaluate sigmoid across an entire array
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-1.0*z))

def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS, lossType):
    error_history = []
    if lossType == "MSE": 
        for i in range(iterations):
            mse = MSE(W,b,trainingData,trainingLabels,reg)
            error_history.append(mse) 
            if mse <= EPS: 
                break 
            mse_gradient_weights, mse_gradient_biases = gradMSE(W,b,trainingData, trainingLabels,reg)
            
            W -= mse_gradient_weights*alpha 
            b -= mse_gradient_biases*alpha
    elif lossType =="CE":
        for i in range(iterations): 
            ce = crossEntropyLoss(W,b,trainingData,trainingLabels,reg)
        
            error_history.append(ce) 
            if ce <= EPS: 
                break 
            ce_gradient_weights, ce_gradient_biases = gradCE(W,b,trainingData, trainingLabels,reg)
            W -= ce_gradient_weights*alpha 
            b -= ce_gradient_biases*alpha 
      
    return W,b
    
def buildGraph(loss=None):
    #Initialize weight and bias tensors
    tf.set_random_seed(421)

    #Tensors to hold the bias and matrix values 
    W = tf.Variable(tf.truncated_normal(mean= 0.0, shape = (1,784), stddev = 0.5, dtype = tf.float32, name = "weight"))
    b = tf.Variable(1.0, name='biases', dtype=tf.float32)
   
    #Tensors to hold the variables 
    X = tf.placeholder(tf.float32,[None, 784],name='X')
    Y = tf.placeholder(tf.float32, [None, 1], name = 'Y')
    reg = tf.constant(0, tf.float32, name = 'lambda') 
    y_predicted = tf.matmul(X,tf.transpose(W)) + b
    regParamater = ((reg/2) * tf.matmul(W,tf.transpose(W)))
    
    if loss == "MSE":
    #Mean square error formula 
        error = (1/2) * tf.reduce_mean(tf.reduce_mean(tf.squared_difference(y_predicted, Y))) + regParamater 
        optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
        train = optimizer.minimize(loss=error)
        
        return W, b,X, y_predicted, Y, error, train, regParamater 
    
    # Your implementation
    elif loss == "CE":
        
        error = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y,logits=y_predicted,name=None)) + regParamater
     #   3q = tf.sigmoid(y_predicted, name = 'sigmoid') 
      #  p = tf.reduce_sum((-1.0 * y_predicted * tf.log(q)) - ((1.0 - y_predicted) * tf.log(1.0 - q)))
   
        #error = tf.reduce_mean(p) + regParamater
        optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
        train = optimizer.minimize(loss=error) #Adding the cross entropy error
        
        return W, b, X,y_predicted, Y, error, train, regParamater 
    
def trainTensorModel(epochs, inputData, outputLabels,typeError,batch_size): 
    # Initialize session
    W, b, X, y_predicted, Y,error,train, reg = buildGraph(typeError)

    init = tf.global_variables_initializer()
   
    X_flattened_training = inputData.reshape(np.shape(inputData)[0],-1)
    
    sess = tf.InteractiveSession()
    sess.run(init)
    
   
     
    totalError = []
    totalAccuracy = []
    i = 0 
    for step in range(0, epochs):
        instances = np.shape(inputData)[0]
        totalBatches = int(instances / batch_size)
        idx = np.random.permutation(len(inputData))
        X_shuffled,Y_shuffled = X_flattened_training[idx], trainTarget[idx]
        
        i = 0
        for k in range(totalBatches):
            X_batch = X_shuffled[i:(i + batch_size),:] 
            y_batch = Y_shuffled[i:(i + batch_size),:]
            _, err, currentW, currentb, yhat = sess.run([train, error, W, b, y_predicted], feed_dict={X:X_batch, Y:y_batch})
         
            i = i + batch_size
        yhat_sign = tf.sign(tf.sign(yhat)+ 1)
        data_tf = tf.convert_to_tensor(y_batch, np.float32)
        correct_prediction = tf.equal(data_tf, yhat_sign)
        acc = 100 * tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
           
        totalAccuracy.append(acc)
        totalError.append(err[0][0])  
        
    print("Final Error", str(err[0][0])) 
    print("Final Accuracy", str(acc)) 
    
    return totalError, sess.run([totalAccuracy])
    
def figPlot(figureNumber,array, title,yLabel): 
    f = plt.figure(figureNumber)
    plt.xlabel('Iterations')
    title = title 
    plt.title(title)  
    plt.ylabel(yLabel)
    labels =[]
    for eachValue, eachData in array.items(): 
        plt.plot(eachData) 
        labels.append(eachValue) 
      
    plt.legend(labels, ncol = len(labels))
    f.show()
    plt.savefig(str(figureNumber))
    
    
    
#============================Helper functions for Part 1 and 2===========================================
def meanSquareError_normalWeights(W,X,y): 
    Total *= (1/(2*N)) 
    Total += (reg/2) * np.matmul(W, np.transpose(W))
    return Total 

def meanSquareError(W,x,y): 

    Total *= (1/(2*N)) 
    Total += (reg/2) * np.matmul(W, np.transpose(W))
    return Total[0][0]

def meanSquareError_normalWeights(W,X,y): 

    #This function returns the minimized weights for the mean square 
    x = X.reshape(np.shape(X)[0],-1)
    biases = np.ones((np.shape(y)))
    
    x = np.concatenate((biases,x),axis=1)
    x_t = np.transpose(x)
    
    x_inverted = np.linalg.inv(np.matmul(x_t,x))
    wlin = np.matmul(np.matmul(x_inverted,x_t),y)
    return wlin[:][1:],wlin[0][0]

def getSign(number): 
    #This function will return the sign of a number 
    if number >= 0: 
        return 1 
    else: 
        return 0

    
def accuracy(x,W,b,y): 
    y_hat = np.matmul(W,np.transpose(x))
    accuracy = 0 
    N =np.shape(y)[0]
    
    for i in range(N): 
        y_hat = getSign(np.matmul(W,np.transpose(X_sliced)))
        y_expected = y[i]   

        if y_hat_sign == ylabel: 
            accuracy += 1
    return 100*(accuracy/N)

def grad_descent_NormalEquation(W,trainingData,trainingLabels):
   W_t,b = meanSquareError_normalWeights(W,trainingData,trainingLabels)       
   return np.transpose(W_t), b

# =============================================================================
    
#error_1,accuracy_training = grad_descent(W,1,trainData,trainTarget,0.0001,5000,0,1e-6)
#error_2,accuracy_training = grad_descent(W,1,trainData,trainTarget,0.001,5000,0,1e-6)
#error_3,accuracy_training = grad_descent(W,1,trainData,trainTarget,0.005,5000,0,1e-6)
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
#figPlot(3,{"Training" : accuracy_training}, "Error" ,"Error") 
#figPlot(4,{"LR = 0.0001" : error_1,"LR = 0.001" : error_2,"LR = 0.005" : error_3}, "Training Data Error With Varying Learning Rates" ,"Error") 
error_train1, accuracy_train1 = trainTensorModel(7,trainData, trainTarget,"MSE",500)
error_train2, accuracy_train2 = trainTensorModel(7,trainData, trainTarget,"MSE",500)
#error_train3, accuracy_train3 = trainTensorModel(700,trainData, trainTarget,"MSE",500)

figPlot(1,{"BS=100" : error_train1, "BS=700" : error_train2},"Error With Varying Batch Size","Error") #Creating a function to map out all of the graphs 
figPlot(2,{"BS=100" : accuracy_train1[0], "BS=700" : accuracy_train2[0]},"Accuracy With Varying Batch Size","Accuracy") #Creating a function to map out all of the graphs 

#gradCE(W,1,trainData, trainTarget, 0)
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

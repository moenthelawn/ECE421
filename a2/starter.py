import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest


def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target

def relu(x):
    # TODO
    return (x * (x > 0))
def gradRelu(x):
    # TODO
    return (1 * (x > 0))
# correct solution:
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=0) # only difference

def computeLayer(X, W, b):
    # TODO
    return (np.matmul(np.transpose(W),X) + b)
    

def CE(target, prediction):
    return -1*np.mean((np.matmul(target, np.log(prediction + 1e-12))))
        
    # TODO

def gradCE(target, prediction):
    return np.mean((target/prediction))
    # TODO


######Part 1 

def forwardProp_predicted(x,W_H,b_h,W_O,b_O,target): 
    s1 = np.matmul(np.transpose(W_H),x) + b_h #Computing the weight matrix between the 
    A1 = relu(s1)
    
    Z2 = np.dot(np.transpose(W_O),A1) + b_O
    A2 = softmax(Z2)
    return np.transpose(A2)


def gradients(x,W_H,b_H,W_O,b_O,prediction,target): 
    z = np.matmul(np.transpose(W_H),x) + b_H 
    s1 = relu(z)
    s1_gradient = gradRelu(z)
    
    er = np.transpose(prediction) - target
    m = 784
    dWO = (1/m)*np.transpose(np.matmul(er,np.transpose(s1)))
    dbO = np.sum(er, axis=0, keepdims=True)
    Z1 = np.multiply(np.matmul(W_O, er), s1_gradient)
    dWH =  (1/m)*np.transpose(np.matmul(Z1, np.transpose(x)))
    dbH = np.sum(Z1, axis=0, keepdims=True)
    return dWO,dbO,dWH,dbH

def gradient_WOutput(error,z1): 
    
    w = np.dot(error,np.transpose((z1)))
    return error* w 
def gradient_bOutput(error, b):
    return error*b
def accuracy(prediction,target): 
    prediction_arg  = prediction.argmax(axis = 1) 
    target_arg = target.argmax(axis=1)
    total = np.in1d(target_arg, prediction_arg) 
    m = np.sum(total[True]) 
    
    return 100* m/np.size(total)
    
def trainNeuralNetwork(data, target,hiddenLayers,epochs) :
    x = np.transpose(data.reshape(np.shape(data)[0],-1))
    nx = np.shape(x)[0] #The number of classes where x is a 10000x784 vector 
    nh = hiddenLayers 
    ny = np.shape(target)[1] #Ten in this case 
    accuracy_total = []
    variance = np.size(2/(1 + nx + ny))  # Here we add a 1 term to account for the input bias 
  
    #Initializing the hidden layer weights 
    W_H = np.random.normal(0,variance,(nx,nh))*0.01 #Matrix to hold the gradients wrt weight
   # W_H = np.zeros((nx,nh))
    b_H = np.transpose(np.ones((1, nh)))
    
    #Initializing the output layer weights 
    
    W_O = np.random.normal(0,variance,(nh,ny))*0.01 #Matrix to hold the gradients wrt weights 
    #W_O = np.zeros((nh,ny))
    b_O = np.transpose(np.ones(shape=(1,ny)))
    v_Whidden = np.full(np.shape(W_H), 1e-5, dtype=float) #Iniatizing the hidden layer 
    v_Woutput = np.full(np.shape(W_O), 1e-5, dtype=float) #Iniatizing the hidden layer 
    
    alpha = 0.005 
    
    for each in range(epochs): 
       predicted = forwardProp_predicted(x,W_H,b_H,W_O,b_O,target) 
       errorCE = CE(target,np.transpose(predicted))
       predicted_t = np.transpose(predicted)
       
      # accuracy = ((np.dot(target,predicted_t) + np.dot(1-target,1-predicted_t))/float(target.size)*100) 
       acc = accuracy(predicted,target)  
       accuracy_total.append(acc) 
       print(acc,errorCE)
       
       dWO,dbO,dWH,dbH = gradients(x,W_H,b_H,W_O,b_O,predicted,np.transpose(target))
           
       #Update the gradients 
       v_Whidden = 0.99*v_Whidden + alpha * dWH #Hidden layer weights 
       v_Woutput = 0.99*v_Woutput + alpha * dWO #Hidden layer biases 
       #accuracy.append(accuracy())
       #v_Bhidden = 0.99*v_Bhidden + alpha * dbH
       #v_Boutput = 0.99*v_Boutput + alpha * dbO 
       
       W_H = W_H - alpha*v_Whidden 
       W_O = W_O - alpha*v_Woutput  
       
       b_H = b_H - alpha*dbH
       b_O = b_O - alpha*dbO
    return accuracy_total



#####Part 2
############################### Building CNN And Running it #####################################

def buildAndRunCNN(trainData, trainTarget, validData, validTarget, testData, testTarget): 
    #Tensor for the images 
    x = tf.placeholder(tf.float32, shape=[None, 28*28], name='x')
    #Tensor holding the true label of each letter image 
    y_true_label = tf.placeholder(tf.float32, shape=[None,10], name='y_true_label')
    y_true_class = tf.argmax(y_true_label, dimension=1)
    
    
    #For Dropout 
    #keep_prob = tf.placeholder(tf.float32)
    
    #1. Input layer 
    # Reshape the image into [num_images, img_height, img_width, num_channels]
    x_image = tf.reshape(x, [-1, 28, 28, 1])
   
   #2. 3x3 Convolutional Layer 32 filters, horizontal and vertical stride of 1. 
    shapeConvLayer = [3, 3, 1, 32]
    with tf.variable_scope("buildCNN", reuse=tf.AUTO_REUSE):
        weightsConv = tf.get_variable(name="weightsConv",shape=shapeConvLayer, 
                                      dtype=tf.float32,
                                      initializer=tf.contrib.layers.xavier_initializer())
        biasesConvInit = tf.constant(1.0, shape=[32])
        biasesConv = tf.get_variable(name="biasesConv", initializer=biasesConvInit)
    convLayer = tf.nn.conv2d(input=x_image, filter=weightsConv, strides=[1,1,1,1],padding="SAME")
    convLayer += biasesConv
    
    #3.  ReLU activation 
    firstReluLayer = tf.nn.relu(convLayer, name="firstReluLayer")
    
    #4. A Batch normalization Layer 
    mean, variance = tf.nn.moments(x=firstReluLayer, axes=[0], keep_dims=True)
    batchNormalizationLayer = tf.nn.batch_normalization(x=firstReluLayer, mean=mean, 
                                                        variance=variance,offset=0.05,
                                                        scale=None,variance_epsilon=0.0001)
    
    #5.A 2x2 max pooling layer 
    maxPoolLayer1 =tf.nn.max_pool(value=batchNormalizationLayer, ksize=[1,2,2,1], strides=[1,1,1,1], padding="SAME")
    
    #6. Flatten layer
    numFeatures = maxPoolLayer1.get_shape()[1:4].num_elements() 
    flattenedLayer = tf.reshape(maxPoolLayer1, [-1, numFeatures])
    
    #7. Fully Connected Layer with 784 Output Units
    shapeFCLayer1 = [numFeatures,784]
    with tf.variable_scope("buildAndRunCNN", reuse=tf.AUTO_REUSE):
        weightsFC = tf.get_variable("weightsFC", shapeFCLayer1, 
                                    initializer=tf.contrib.layers.xavier_initializer())
        biasesFCInit = tf.Variable(tf.constant(1.0, shape=[784]))
        biasesFC = tf.get_variable("biases_fc", initializer=biasesFCInit)
    fullyConnectedLayer1 = tf.matmul(flattenedLayer, weightsFC) + biasesFC
    
    # DROPOUT 
    #dropoutLayer= tf.nn.dropout(fullyConnectedLayer1, keep_prob)
    
    #8. ReLU activation
    secondReluLayer = tf.nn.relu(fullyConnectedLayer1, name="secondReluLayer")
    
    #9. fully connected layer with 10 output units 
    shapeFCLayer2 = [784,10]
    with tf.variable_scope("buildAndRunCNN", reuse=tf.AUTO_REUSE):
        weightsFC2 = tf.get_variable("weightsFC2", shapeFCLayer2, 
                                     initializer=tf.contrib.layers.xavier_initializer())
        biasesFC2Init = tf.Variable(tf.constant(1.0, shape=[10]))
        biasesFC2 = tf.get_variable("biases_fc2", initializer=biasesFC2Init)
    fullyConnectedLayer2 = tf.matmul(secondReluLayer, weightsFC2) + biasesFC2
        
    #10. Softmax Output 
    with tf.variable_scope("Softmax"):
        y_pred = tf.nn.softmax(fullyConnectedLayer2)
        y_pred_class = tf.argmax(y_pred, dimension=1)
    
    #11. Cross entropy of output to compute the loss
    with tf.name_scope("cross_ent"):
        beta = tf.Variable(tf.constant(0.5, name="beta"))
        crossEntropy = tf.nn.softmax_cross_entropy_with_logits(logits=fullyConnectedLayer2, 
                                                               labels=y_true_label)
        loss = tf.reduce_mean(crossEntropy)
        #Using L2 Regularization -- Not complete yet 
        regularizers = tf.norm(weightsConv) + tf.norm(weightsFC) + tf.norm(weightsFC2)
        loss = loss + tf.reduce_mean((beta/2) * regularizers)
        
        
    # Use=ing Adam Optimizer
    with tf.variable_scope("buildAndRunCNN",reuse=tf.AUTO_REUSE):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

    # Measuring the Accuracy
    with tf.name_scope("accuracy"):
        correctPrediction = tf.equal(y_pred_class, y_true_class)
        accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
   
    
    ###################    Running the CNN Now using SGD #############################3
    
    X_flattened_training = trainData.reshape(np.shape(trainData)[0],-1)
    X_flattened_validation = validData.reshape(np.shape(validData)[0],-1)
    X_flattened_testing = testData.reshape(np.shape(testData)[0],-1)
    
    epochs = 50 
    batch_size = 32
    trainingLoss=[]
    validationLoss=[]
    testLoss=[]
    accuraciesTraining=[]
    accuraciesValidation=[]
    accuraciesTest = []
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        #Looping over the number of epochs
        for epoch in range(epochs):
            instances = np.shape(trainData)[0]
            totalBatches = int(instances / batch_size)
            idx = np.random.permutation(len(trainData))
            X_shuffled,Y_shuffled = X_flattened_training[idx], trainTarget[idx]

            i=0
            train_accuracy = 0
            train_loss = 0 
            validation_accuracy = 0 
            validation_loss = 0
            test_accuracy = 0 
            test_loss = 0
            #Loop over the batch size 
            for k in range(totalBatches):
                X_batch = X_shuffled[i:(i + batch_size),:] 
                Y_true_batch = Y_shuffled[i:(i + batch_size),:]
                feed_dict_train = {x: X_batch, y_true_label: Y_true_batch}
                sess.run(optimizer, feed_dict=feed_dict_train)
                #Training
                train_accuracy += sess.run(accuracy, feed_dict=feed_dict_train)
                train_loss = sess.run(loss, feed_dict=feed_dict_train)
                trainingLoss.append(train_loss)
                i = i + batch_size
            
            #Training
            train_accuracy /= int(len(trainData)/batch_size)
            train_accuracy = train_accuracy * 100
            accuraciesTraining.append(train_accuracy)
            trainingLoss.append(train_loss)
            
            #Validation Data
            validation_accuracy, validation_loss = sess.run([accuracy, loss], 
                                                            feed_dict={x:X_flattened_validation, 
                                                                       y_true_label:validTarget})
            validation_accuracy = validation_accuracy *100
            accuraciesValidation.append(validation_accuracy)
            validationLoss.append(validation_loss)
            
            #Test Data
            test_accuracy, test_loss = sess.run([accuracy, loss], 
                                                feed_dict={x:X_flattened_testing, 
                                                           y_true_label:testTarget})
            test_accuracy = test_accuracy *100 
            accuraciesTest.append(test_accuracy) 
            testLoss.append(test_loss)
 
            print("Epoch",epoch," Complete")
        
        
        #After the tf Session is completed, we plot the training, validation and test accuracy and losses 
        figPlot(1,{"Training Cross Entropy Loss":trainingLoss},"Training Loss","Cross Entropy Loss","Loss Per Training Batch" )
        figPlot(2,{"Training Accuracy": accuraciesTraining}, "Training Accuracy Per Epoch", "Percentage","Epoch")
        
        
        figPlot(3,{"Validation Accuracy":accuraciesValidation}, "Validation Accuracy per Epoch","Percentage","Accuracy Per training Batch")
        figPlot(4,{"Validation Cross Entropy Loss": validationLoss}, "Validation Loss Per Epoch","Cross Entropy Loss", "Loss")
        
        figPlot(5,{"Test Accuracy":accuraciesTest}, "Test Accuracy per Epoch","Percentage","Epoch")
        figPlot(6, {"Test Cross Entropy Loss": testLoss}, "Test Loss Per Epoch","Cross Entropy Loss", "Loss")
    
    return train_accuracy, validation_accuracy, test_accuracy
   
    
#Helper function to plot the different stuff, array is a dictionary type
def figPlot(figureNumber,array, title,yLabel, xLabel): 
    f = plt.figure(figureNumber)
    plt.xlabel(xLabel)
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


# Load Data, One hot Encode the target labels, and then Build and Run the CNN !
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
newTrainTarget, newValidTarget, newTestTarget = convertOneHot(trainTarget,validTarget,testTarget)

##Running part 1 #####
#acc_500 = trainNeuralNetwork(trainData, trainTarget_onehot,500,200)
acc_1000_train = trainNeuralNetwork(trainData, newTrainTarget,1000,200)
acc_1000_valid = trainNeuralNetwork(validData, newValidTarget,1000,200)
acc_1000_test = trainNeuralNetwork(testData, newTestTarget,1000,200)
#acc_2000 = trainNeuralNetwork(trainData, trainTarget_onehot,2000,200)


## Running Part 2#####

train_accuracy, validation_accuracy, test_accuracy = buildAndRunCNN(trainData, newTrainTarget, validData, newValidTarget, testData, newTestTarget)

print("Final Training Accuracy", train_accuracy)
print("Final Validation Accuracy", validation_accuracy)
print("Final Test Accuracy", test_accuracy)



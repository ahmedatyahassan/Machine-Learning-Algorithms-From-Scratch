import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Get the data
df = pd.read_csv('heart.csv')

features = ['trestbps', 'chol', 'thalach', 'oldpeak']
value = ['target']

# Separate training data
x1Train = df.loc[0:100, features]
y1Train = df.loc[0:100, value]

x2Train = df.loc[202:302, features]
y2Train = df.loc[202:302, value]

xframesTrain = [x1Train, x2Train]
xTrain = pd.concat(xframesTrain)

yframesTrain = [y1Train, y2Train]
yTrain = pd.concat(yframesTrain)

# Separate testing data
xTest = df.loc[100:202, features]
yTest = df.loc[100:202, value]

# Feature Scaling
xTrain = (xTrain - xTrain.mean()) / xTrain.std()
xTest = (xTest - xTest.mean()) / xTest.std()

# Adding x0 values
xTrain.insert(0, 'X0', 1)
xTest.insert(0, 'X0', 1)

# Convert Data Into arrays && Set Parameters(theta)
xTrain = xTrain.values
yTrain = yTrain.values.flatten()

theta = np.array([0, 0, 0, 0, 0])

xTest = xTest.values
yTest = yTest.values.flatten()


#Cost Function
def compute_cost(segmoid, y):
    m = xTrain.shape[0]

    segmoid = np.where(segmoid == 1, 0.99999999, segmoid)
    segmoid = np.where(segmoid == 0, 0.00000001, segmoid)

    cost = -(1 / m) * np.sum(np.dot(np.log(segmoid), y) + np.dot(np.log(1 - segmoid), (1 - y)))
    return cost


#Accuracy Function
def accuracy(segmoid, y, temp):
    corAns = 0

    for i in range(len(segmoid)):
        if segmoid[i] < 0.5:
            if y[i] == 0:
                corAns = corAns + 1
        if segmoid[i] >= 0.5:
            if y[i] == 1:
                corAns = corAns + 1

    if temp == 0:
        print("Accuracy of Training is : ", (corAns / (y.shape[0])) * 100, "%")
        print("==========================================================================")
    else:
        print("Accuracy of Testing is : ", (corAns / (y.shape[0])) * 100, "%")
        print("==========================================================================")

#Gradient Descent Function
def gradient_descent(x, y, theta, alpha, iterations):

    cost = np.zeros(iterations)
    for i in range(iterations):

        segmoid = 1 / (1 + np.exp(-np.dot(x, theta)))
        error = segmoid - y

        term = (1/len(x)) * (x.transpose().dot(error))
        theta = theta - alpha * term

        cost[i] = compute_cost(segmoid, y)

    accuracy(segmoid, y, 0)

    return theta, cost

def segmoid_test(x, y, theta):
    segmoid = 1 / (1 + np.exp(-np.dot(x, theta)))
    accuracy(segmoid, y, 1)
    for i in range (len(segmoid)):
        if segmoid[i] < 0.5:
            segmoid[i] = 0
        else:
            segmoid[i] = 1
    print("Predicted values :", segmoid)

iters = 12000
list = []

alpha = [0.001,0.003,0.01,0.03,0.1,0.3,1.0]
for i in range(len(alpha)):
    print("At Learning Rate", alpha[i])
    print("----------------------")
    bestTheta, cost = gradient_descent(xTrain, yTrain, theta, alpha[i], iters)
    print("Best Theta: ", bestTheta)
    print("==========================================================================")
    print("The costs of interations: ")
    print(cost)
    print("==========================================================================")
    segmoid_test(xTest, yTest, bestTheta)
    list.clear()
    for i in range(iters):
        list.append(i)
    plt.scatter(list, cost)
    plt.xlabel('Iteration')
    plt.ylabel('cost')
    plt.title("Error")
    plt.show()
    print("\n\n")

import numpy as np
import pandas as pd

# Get the data
data = pd.read_csv('heart.csv')

features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca','thal']
# features = ['trestbps', 'chol', 'thalach', 'oldpeak']
# features = ['fbs', 'restecg', 'thalach', 'exang', 'oldpeak']
# features = ['age', 'sex', 'cp', 'trestbps', 'exang', 'oldpeak', 'slope']
value = ['target']

# Randomize data
setTrain = data.sample(n=220)
xTrain = setTrain[features]
yTrain = setTrain[value]
setTest = data.sample(n=83)
xTest = setTest[features]
yTest = setTest[value]

# Feature Scaling
xTrain = (xTrain - xTrain.mean()) / xTrain.std()
xTest = (xTest - xTest.mean()) / xTest.std()

# Adding x0 values
xTrain.insert(0, 'X0', 1)
xTest.insert(0, 'X0', 1)

# Convert Data Into arrays
xTrain = xTrain.values
yTrain = yTrain.values.flatten()

xTest = xTest.values
yTest = yTest.values.flatten()

# Convert each (zero) into (-1) in the actual y
yTrain = np.where(yTrain == 0, -1, yTrain)
yTest = np.where(yTest == 0, -1, yTest)

#Accuracy Function
def accuracy(yPredict, y, temp):
    corAns = 0

    for i in range(len(y)):
        if y[i] == yPredict[i]:
            corAns += 1

    if temp == 0:
        print("Accuracy of Training is : ", (corAns / (y.shape[0])) * 100, "%")
        # print("==========================================================================")
    else:
        print("Accuracy of Testing is : ", (corAns / (y.shape[0])) * 100, "%")
        print("==========================================================================")


def predict(x, w):
    y = np.dot(x, w)
    for i in range(len(y)):
        if y[i] > 0:
            y[i] = 1
        else:
            y[i] = -1
    return y

def gradientDescent(x, y, iterations = 3000, learningRate = 0.003):
    # fill the weight array by zeros
    w = np.zeros(len(x[0]))
    lmda = 1 / iterations
    # Training
    # print("starts training")
    cost_history = []
    for i in range(iterations):
        cost = 0
        for j in range(x.shape[0]):
            temp = y[j] * np.dot(x[j], w)
            if temp >= 1:
                w -= learningRate * 2 * lmda * w
            else:
                w += learningRate * (np.dot(x[j], y[j]) - 2 * lmda * w)
                cost += (1 - temp)

        cost_history.append(cost)

    return w, cost_history

iters = 1000
alpha = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]

for i in range(len(alpha)):
    print("At Learning Rate", alpha[i])
    print("----------------------")

    w, costHistory = gradientDescent(xTrain, yTrain, iters, alpha[i])

    yPredictTrain = predict(xTrain, w)
    accuracy(yPredictTrain, yTrain, 0)
    # print("Best Weight: ", w)
    # print("==========================================================================")
    # print("The costs of interations: ")
    # print(costHistory)
    # print("==========================================================================")
    yPredictTest = predict(xTest, w)
    accuracy(yPredictTest, yTest, 1)
    print("\n\n")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None


def perceptron(mergedTrainData, mergedTestData):
    trainedWeights, trainedBias, iterations = perceptronTrain(mergedTrainData, 300, .3)
    results = perceptronTest(mergedTestData, trainedWeights, trainedBias)
    print(f"HINGE LOSS: {results[1]}")
    print(f"Accuracy: {results[0]}")
    #print(f"Num Iterations: {iterations}")
    hingeLoss = results[1]
    accuarcy = results[0]
    return hingeLoss, accuarcy, iterations

def perceptronTrain(mergedData, MaxIter, learningRate):
    weights = np.zeros(len(mergedData.iloc[0]) - 1)
    bestWeights = np.zeros(len(mergedData.iloc[0]) - 1)
    bestBias = 0
    predictedLabels = np.array([0 for i in range(len(mergedData.index))])
    label = np.array(list(mergedData['survived']))
    bias = 0
    numRows = len(mergedData.index)
    iteration = 0
    accuracy = 0
    # and not(label == predictedLabels).all()
    while iteration < MaxIter:
        for i in range(numRows):
            # compute activation
            currRow = mergedData.iloc[i]
            activation = 0
            for n in range(len(currRow) - 1):
                activation += weights[n] * currRow[n]
        
            activation += bias
            predicted = -1
            if activation > 0:
                predicted = 1
            predictedLabels[i] = predicted
            # if ya <= 0
            if currRow['survived'] * activation <= 0:
                for n in range(len(currRow) - 1):
                    weights[n] = weights[n] + learningRate * (currRow['survived'] * currRow[n]) 
                bias = bias + currRow['survived'] * learningRate 

        # print((label == predictedLabels).sum() * 1.0 / len(predictedLabels))
        if (label == predictedLabels).sum() * 1.0 / len(predictedLabels) > accuracy:
            accuracy = (label == predictedLabels).sum() * 1.0 / len(predictedLabels)
            bestWeights = weights
            bestBias = bias
        iteration += 1
  
    return bestWeights, bestBias, iteration


def perceptronTest(mergedTestData, trainedWeights, trainedBias):
    numRows = len(mergedTestData.index)
    count = 0
    # hingeLoss = 0
    for i in range(numRows):
        currRow = mergedTestData.iloc[i]
        activation = 0
        for n in range(len(currRow) - 1):
            activation += trainedWeights[n] * currRow[n]
        # currHingeLoss = max(0, 1 - activation * currRow['survived'])
        # hingeLoss += currHingeLoss

        activation += trainedBias
        predictedLabel = 1
        if activation <= 0:
            predictedLabel = -1
        if predictedLabel == currRow['survived']:
            count += 1

    hinge = hingeLoss(trainedWeights, mergedTestData)  
    return [count * 1.0 / numRows, hinge]

def hingeLoss(trainedWeights, test):
    
    numRows = len(test.index)
    # print(numRows)
    hingeLoss = 0
    for i in range(numRows):
        currRow = test.iloc[i]

        activation = 0
        for i in range(len(currRow) - 1):
            activation += trainedWeights[i] * currRow[i]
        
        prediction = -1
        if activation > 0:
            prediction = 1
        hingeLoss += max(0, 1 - prediction * currRow['survived'])
        
    return hingeLoss / numRows

def discretize(mergedData):
    df = pd.DataFrame()
    df["discrete Age"] = pd.qcut(mergedData['Age'], q=2, duplicates='drop')
    df["discrete Relatives"] = pd.qcut(mergedData['relatives'], q=2, duplicates='drop')
    df['discrete Fare'] = pd.qcut(mergedData['Fare'], q=2, duplicates='drop')

    for i in range(len(mergedData.index)):
        if  df['discrete Age'][i].left <= mergedData['Age'][i] < df['discrete Age'][i].right:
            mergedData['Age'][i] = 0
        else:
            mergedData['Age'][i] = 1
     
    for i in range(len(mergedData.index)):
        if  df['discrete Relatives'][i].left <= mergedData['relatives'][i] < df['discrete Relatives'][i].right:
            mergedData['relatives'][i] = 0
        else:
            mergedData['relatives'][i] = 1
     
    for i in range(len(mergedData.index)):
        if  df['discrete Fare'][i].left <= mergedData['Fare'][i] < df['discrete Fare'][i].right:
            mergedData['Fare'][i] = 0
        else:
            mergedData['Fare'][i] = 1
    return mergedData


def evaluation(mergedData):
    percentages = [1,10,50]
    averageHingeLossError = []
    averageDefaultError = []
    
    for percent in percentages:
        hingeLossList = []
        iterationsList = []
        defaultErrorList = []
        accuracyList = []
        for i in range(10):
            data = mergedData.copy(deep=False)
            fraction = percent * 1.0 / 100
            trainData = data.sample(frac=fraction)
            defaultTrainData = trainData.copy(deep=False)
            rowIndexList = []
            
            rowIndexList = trainData.index.values.tolist()
            data = data.drop(rowIndexList)
            
            testdata = data
            # print(len(testdata.index))
            defaultResults = defaultAccuracy(defaultTrainData, testdata)
            defaultErrorList.append(defaultResults[0])

            hingeLoss, accuracy, iterations = perceptron(trainData, testdata)
            accuracyList.append(accuracy)
            iterationsList.append(iterations)
            hingeLossList.append(hingeLoss)

        averageDefaultError.append(sum(defaultErrorList) * 1.0 / 10)
        averageHingeLossError.append(sum(hingeLossList) * 1.0 / 10)

        print(f'Percent of overall Data used to train Perceptron: {percent}%')
        print(f'Average Default Error: {sum(defaultErrorList) * 1.0 / 10}')
        print(f'Average Hinge Loss: {sum(hingeLossList) * 1.0 / 10}')
        print(f'Average Accuracy: {sum(accuracyList) * 1.0 / 10}')
        print(f'Average Number of Iterations: {sum(iterationsList) * 1.0 / 10}')
        print()


    plt.plot(percentages, averageDefaultError, label = 'Default Error')
    plt.plot(percentages, averageHingeLossError, label = 'Hinge Loss')
    plt.title("Training Size(%) vs Average Hinge Loss")
    plt.ylabel("Average Hinge Loss")
    plt.xlabel("Training Size(%)")
    plt.legend()
    plt.savefig("hingeLoss")
    plt.clf()

def mostCommonLabel(train_label):
    numZeros = 0
    numOnes = 0
    for i in train_label['survived']:
        if i == 1:
            numOnes += 1
        else:
            numZeros += 1
    
    mostCommon = -1
    if numOnes > numZeros:
        mostCommon = 1
    return mostCommon

def defaultAccuracy(mergedData, mergedTestData):
    count = 0
    copyMergeData = mergedData.copy(deep=False)
    pLabel = mostCommonLabel(mergedData)
    trainedWeights, trainedBias, iterations = perceptronTrain(mergedTrainData, 77, .3)
    hinge = hingeLoss(trainedWeights, mergedTestData)
    
    # print(f"ZERO ONE LOSS: {zeroOneLoss}, Accuracy: {accuracy}")
    return [hinge]


def replaceSurvived(survivedCol):
    
    for i in range(len(survivedCol)):
        if survivedCol[i] == 0:
            survivedCol[i] = -1
        else:
            survivedCol[i] = 1

    return survivedCol

def cleanData(data):
    columns = list(data)
    for col in columns:
        values = data[col].value_counts().index.tolist()
        # print(values[0])
        data[col] = data[col].replace(np.nan, values[0])
    return data


def mergeData(data, label):
   return pd.concat([data, label], axis=1) 

# Part 2


if __name__ == "__main__":
    import sys
    trainDataPath = sys.argv[1]
    trainLabelPath = sys.argv[2]

    train_data = pd.read_csv(trainDataPath)
    train_label = pd.read_csv(trainLabelPath)

    train_data = cleanData(train_data)
    mergedTrainData = mergeData(train_data, train_label)
    mergedTrainData = discretize(mergedTrainData)
    

    testDataPath = sys.argv[3]
    testLabelPath = sys.argv[4]
    test_data = pd.read_csv(testDataPath)
    test_label = pd.read_csv(testLabelPath)
    
    test_data = cleanData(test_data)
    mergedTestData = mergeData(test_data, test_label)
    mergedTestData = discretize(mergedTestData)

    mergedTrainData['survived'] = replaceSurvived(mergedTrainData['survived'])
    # print(mergedTrainData.to_string())


    mergedTestData['survived'] = replaceSurvived(mergedTestData['survived'])
    # print(mergedTestData.to_string())
    
    perceptron(mergedTrainData, mergedTestData)
    # evaluation(mergedTrainData)